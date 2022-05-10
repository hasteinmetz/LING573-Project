#!/usr/bin/env python

''' References:
	- https://www.kaggle.com/code/ynouri/random-forest-k-fold-cross-validation/notebook
'''
import utils
import torch
import argparse
import sys
import numpy as np
import pandas as pd
from typing import *
from featurizer import featurize, TFIDFGenerator
from torch.utils.data import DataLoader
from finetune_dataset import FineTuneDataSet
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from fine_tune import *
from math import ceil

nn = torch.nn

class Ensemble():
	def __init__(self, tokenizer, steps) -> None:
		'''
		arguments:
			- logreg_config: filepath to .json config specifying parameters for logistic regression classifier
			- train_lex_feat: training lexical feature instances
			- train_labels: corresponding labels for training lexical feature instances
			
		sets up ensemble model.  expects roberta model to be pretrained. expects random forest and logistic 
		regression models to require training
		'''
		super().__init__()
		self.roberta_model = RobertaModelWrapper(32, 0.00005, tokenizer, steps)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		self.roberta_model.resize_token_embeddings(len(self.roberta_tokenizer))
		self.rf_config = utils.load_json_config(forest_config_path)
		self.random_forest = None
		if 'src/model' in self.roberta_config._name_or_path:
			self.pretrained = 'fine-tuned'
		else:
			self.pretrained = 'roberta-base'

		logreg_config = utils.load_json_config(logreg_config_path)
		self.classifier = LogisticRegression(penalty=logreg_config["penalty"], random_state=logreg_config["random_state"],\
			 solver=logreg_config["solver"], verbose=logreg_config["verbose"])
	
	def train_random_forest(self, lexical_features: np.ndarray, labels: np.ndarray) -> None:
		'''
		arguments:
			- config: grid of parameter options to configure rf model with 
			- lexical_features: array of lexical features, one row per sample in data
			- labels: corresponding classification label for each sample
		returns:
			sklearn.ensemble.RandomForestClassifier

		uses cross-fold validation to train rf classifier with hyperparameters designated in config.
		'''
		self.random_forest = RandomForestClassifier(n_estimators=self.rf_config["n_estimators"], bootstrap=self.rf_config["bootstrap"],\
			 max_depth=self.rf_config["max_depth"], max_features=self.rf_config["max_features"],\
				  min_samples_leaf=self.rf_config["min_samples_leaf"], min_samples_split=self.rf_config["min_samples_split"],\
					  n_jobs=-1)
		cross_validation_trainer = StratifiedKFold(n_splits=9, random_state=42, shuffle=True)
		accuracy = {"train": [], "test": []}
		for (train_idx, test_idx), i in zip(cross_validation_trainer.split(lexical_features, labels), range(9)):
			self.random_forest.fit(lexical_features[np.asarray(train_idx)], labels[np.asarray(train_idx)])
			accuracy["train"].append(self.random_forest.score(lexical_features[train_idx], labels[train_idx]))
			accuracy["test"].append(self.random_forest.score(lexical_features[test_idx], labels[test_idx]))
		accuracy_df = pd.DataFrame(accuracy)
		print("random forest classifier accuracy")
		print(accuracy_df)


class Ensemble(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, output_size: int):
		super(Ensemble, self).__init__()
		self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size+1)
		)
		self.output = nn.Linear(2+2, output_size)

	def forward(self, data: dict, sentences: List[str], featurizer: Callable, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		outputs_roberta = self.roberta(**inputs).logits
		features_tensor = torch.tensor(featurizer(sentences), dtype=torch.float).to(device)
		outputs_mlp = self.mlp(features_tensor)
		classifier_in = torch.cat((outputs_roberta, outputs_mlp), axis=1)
		logits = self.output(classifier_in)
		return logits

def train(model: Ensemble, sentences: List[str], labels: List[str], epochs: int, batch_size: int, lr: int,
	featurizer: Callable, tokenizer: RobertaTokenizer, optimizer: torch.optim, loss_fn: Callable, device: str):
	'''Train the Ensemble neural network'''
	model.to(device)
	model.train()
	optim = optimizer(model.parameters(), lr=lr, weight_decay=1e-5)

	metrics = []
	for metric in ['f1', 'accuracy']:
		m = load_metric(metric)
		metrics.append(m)

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)

	dataset = FineTuneDataSet(shuffled_sentences, shuffled_labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	for epoch in range(epochs):
		for i, (batch, X) in enumerate(dataloader):

			y = torch.reshape(batch['labels'], (batch['labels'].size()[0], 1)).float().to(device)

			optim.zero_grad()

			output = model(batch, X, featurizer, device)

			loss = loss_fn(output, y)
			loss.backward()
			optim.step()

			# add batched results to metrics
			pred_argmax = torch.round(torch.sigmoid(output))
			for m in metrics:
				m.add_batch(predictions=pred_argmax, references=y)
        
			if (i + 1) % 6 == 0:
		
				# output metrics to standard output
				print(f'({epoch}, {(i + 1) * batch_size}) Loss: {loss.item()}', file = sys.stderr)

		# output metrics to standard output
		values = f"" # empty string 
		for m in metrics:
			val = m.compute()
			values += f"{m.name}:\n\t {val}\n"
		print(values, file = sys.stderr)


def evaluate(model: Ensemble, sentences: List[str], labels: List[str], batch_size: int, lr: int, 
	tokenizer: RobertaTokenizer, featurizer: Callable, device: str):
	'''Train the Ensemble neural network'''
	model.to(device)
	model.eval()

	metrics = []
	for metric in ['f1', 'accuracy']:
		m = load_metric(metric)
		metrics.append(m)

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)

	dataset = FineTuneDataSet(shuffled_sentences, shuffled_labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	predictions = []

	for batch, X in dataloader:

		y = torch.reshape(batch['labels'], (batch['labels'].size()[0], 1)).float().to(device)

		output = model(batch, X, featurizer, device)

		# add batch to output
		pred_argmax = torch.round(torch.sigmoid(output))
		as_list = pred_argmax.clone().detach().to('cpu').tolist()
		predictions.append(as_list)

		# add batched results to metrics
		for m in metrics:
			m.add_batch(predictions=pred_argmax, references=y)
	
	# output metrics to standard output
	print(f'Loss: {loss.item()}', file = sys.stderr)

		# get roberta predictions
		roberta_class_prob = ensemble.roberta_model.evaluate(meta_roberta_input, ['f1', 'accuracy'], device)

		# combine mlp output and roberta embeddings and feed to logisitical regression model
		print(roberta_class_prob.shape, mlp_class_prob.shape)
		combined_class_prob = np.concatenate((roberta_class_prob, mlp_class_prob), axis=1)
		print(f"(train-fold {i}) training logistic regression classifier", file=sys.stderr)
		ensemble.classifier.partial_fit(combined_class_prob, meta_model_labels, np.unique(meta_model_labels))

		#output training performance
		training_accuracy = ensemble.classifier.score(combined_class_prob, meta_model_labels)
		print("(train-fold {i}) ensemble training accuracy: {}".format(training_accuracy))


def predict(ensemble: Ensemble, dev_sentences: List[str], dev_labels: List[str], device: str, tfidf: TFIDFGenerator) -> np.ndarray:
	# get the feature vectors and embeddings
	dev_data = FineTuneDataSet(dev_sentences, dev_labels)
	dev_lex_feat, dev_roberta_input = get_ensemble_inputs(dev_data, ensemble, tfidf)

	mlp_class_prob = ensemble.mlp.predict_proba(dev_lex_feat)
	roberta_class_prob = ensemble.roberta_model.evaluate(dev_roberta_input, ['f1', 'accuracy'], device)
	combined_class_prob = np.concatenate((roberta_class_prob, mlp_class_prob), axis=1)

	predicted_labels = ensemble.classifier.predict(combined_class_prob)
	predicted_accuracy = ensemble.classifier.score(combined_class_prob, dev_labels)
	print("\tdev accuracy: {}".format(predicted_accuracy))
	return predicted_labels, predicted_accuracy



def main(args: argparse.Namespace) -> None:
	# check if cuda is avaiable
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
		torch.device(device)
		print(f"Using {device} device")
		print(f"Using the GPU:{torch.cuda.get_device_name(0)}")
	else:
		torch.device(device)
		print(f"Using {device} device")

	#load data
	print("loading training and development data...")
	train_sentences, train_labels = utils.read_data_from_file(args.train_data_path)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_data_path)

	# initialize tf-idf vectorizer
	tfidf = TFIDFGenerator(train_sentences, 'english', train_labels)
	featurizer = lambda x: featurize(x, tfidf)

	#get features
	print("preparing hurtlex dictionary...")
	hurtlex_dict, hurtlex_feat_list = utils.read_from_tsv(args.hurtlex_path)
	print("featurizing training and dev data...")
	train_feature_vector = featurize(train_sentences, train_labels, hurtlex_dict, hurtlex_feat_list)
	dev_feature_vector = featurize(dev_sentences, dev_labels, hurtlex_dict, hurtlex_feat_list)

	#get tokenized input
	print("preparing input for roberta model...")
	train_tokenized_input = ensemble_model.roberta_tokenizer(train_sentences, return_tensors="pt", padding=True)
	dev_tokenized_input = ensemble_model.roberta_tokenizer(dev_sentences, return_tensors="pt", padding=True)

	train_dataset = FineTuneDataSet(train_sentences, train_labels)
	dev_dataset = FineTuneDataSet(dev_sentences, dev_labels)
	train_dataset.tokenize_data(ensemble_model.roberta_tokenizer)
	dev_dataset.tokenize_data(ensemble_model.roberta_tokenizer)

	#send to train
	print("training ensemble model...")
	train_ensemble(ensemble_model, train_feature_vector, train_labels, train_dataset, device)

	#run whole ensemble on dev data 
	print("predicting dev labels...")
	dev_predicted_labels, dev_accuracy = predict(ensemble_model, dev_feature_vector, dev_labels, dev_dataset, device)

	#output results
	print("outputting dev classification output...")
	dev_out_d = {'sentence': dev_sentences, 'predicted': dev_predicted_labels, 'correct_label': dev_labels}
	dev_out = pd.DataFrame(dev_out_d)
	dev_out.to_csv(args.output_file, index=False, encoding='utf-8')

	print("outputting results...")
	dev_f1 = f1_score(dev_labels, dev_predicted_labels)
	with open(args.results_file, 'w') as f:
		f.write("accuracy: {}\n".format(dev_accuracy))
		f.write("f1: {}\n".format(dev_f1))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--roberta_config", help="configuration settings for roberta model")
	parser.add_argument("--random_forest_config", help="configuration settings for random forest classifier")
	parser.add_argument("--logistic_regression_config", help="configuration settings for logistic regression classifier")
	parser.add_argument("--train_data_path", help="path to input training data file")
	parser.add_argument("--dev_data_path", help="path to input dev data file")
	parser.add_argument("--hurtlex_path", help="path to hurtlex lexicon file")
	parser.add_argument("--output_file", help="path to output data file")
	parser.add_argument("--results_file", help="path to which accuracy and f1 score will be written to")
	args = parser.parse_args()

	main(args)
