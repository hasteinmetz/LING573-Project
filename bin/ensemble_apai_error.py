#!/usr/bin/env python

''' References:
	- https://www.kaggle.com/code/ynouri/random-forest-k-fold-cross-validation/notebook
'''
import utils
import torch
import argparse
import numpy as np
import pandas as pd
from typing import *
from featurizer import featurize
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from finetune_dataset import FineTuneDataSet
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

nn = torch.nn

class Ensemble():
	def __init__(self, roberta_config_path: str, forest_config_path: str, logreg_config_path: str) -> None:
		'''
		arguments:
			- roberta_config_path: filepath to .json config specifying parameters for roberta model
			- forest_config_path: filepath to .json config specifying parameters for random forest classifier
			- logreg_config_path: filepath to .json config specifying parameters for logistic regression classifier
			- train_lex_feat: training lexical feature instances
			- train_labels: corresponding labels for training lexical feature instances
			
		sets up ensemble model.  expects roberta model to be pretrained. expects random forest and logistic 
		regression models to require training
		'''
		super().__init__()
		self.roberta_config = RobertaConfig.from_json_file(roberta_config_path)
		self.roberta_model = RobertaForSequenceClassification(self.roberta_config)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		self.roberta_model.resize_token_embeddings(len(self.roberta_tokenizer))
		self.rf_config = utils.load_json_config(forest_config_path)
		self.random_forest = None

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


def train_ensemble(ensemble: Ensemble, train_lex_feat: np.ndarray, train_labels: np.ndarray, roberta_input: BatchEncoding, device: str) -> None:
	'''
	arguments:
		- ensemble: ensemble model
		- train_lex_feat: vectorized features
		- train_labels: golden labels for input data
		- roberta_input: tokenized input for roberta
		- device: which device roberta models and data should be run and stored on
	
	trains the random forest classifier on the feature vector, then trains log reg classifier on combined output from the random forest classifier
	and roberta model
	'''
	#train random forest
	print("\ttraining random forest classifier...")
	ensemble.train_random_forest(train_lex_feat, train_labels)
	rf_class_prob = ensemble.random_forest.predict_proba(train_lex_feat)

	#get roberta embeddings
	ensemble.roberta_model.eval()
	ensemble.roberta_model.to(device)
	dl = DataLoader(roberta_input, batch_size=32)

	roberta_class_prob = None 
	for batch in dl:
		batch['labels'] = batch.pop('label')
		inputs = {k: y.to(device) for k,y in batch.items()}

		with torch.no_grad():
			outputs = ensemble.roberta_model(**inputs)
		logits = outputs.logits
		probs = logits.clone().detach().to('cpu').numpy()
		if roberta_class_prob is None:
			roberta_class_prob = probs
		else:
			roberta_class_prob = np.concatenate((roberta_class_prob, probs), axis=0)
	
	#combine rf output and roberta embeddings and feed to logisitical regression model
	combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
	print("\ttraining logistic regression classifier")
	ensemble.classifier.fit(combined_class_prob, train_labels)

	#output training performance
	training_accuracy = ensemble.classifier.score(combined_class_prob, train_labels)
	print("logreg classifier training accuracy: {}".format(training_accuracy))


def predict(ensemble: Ensemble, dev_lex_feat: np.ndarray, dev_labels: np.ndarray, roberta_input: BatchEncoding, device: str) -> Tuple[np.ndarray, float]:
	'''
	arguments:
		- ensemble: ensemble model
		- dev_lex_feat: vectorized features
		- dev_labels: golden labels for input data
		- roberta_input: tokenized input for roberta
		- device: which device roberta models and data should be run and stored on
	
	does a forward pass of the ensemble on the provided input data and returns the accuracy and f1 score metrics
	'''
	rf_class_prob = ensemble.random_forest.predict_proba(dev_lex_feat)

	ensemble.roberta_model.eval()
	ensemble.roberta_model.to(device)
	dl = DataLoader(roberta_input, batch_size=32)

	roberta_class_prob = None
	for batch in dl:
		batch['labels'] = batch.pop('label')
		inputs = {k: y.to(device) for k,y in batch.items()}

		with torch.no_grad():
			outputs = ensemble.roberta_model(**inputs)
		logits = outputs.logits
		probs = logits.clone().detach().to('cpu').numpy()
		if roberta_class_prob is None:
			roberta_class_prob = probs
		else:
			roberta_class_prob = np.concatenate((roberta_class_prob, probs), axis=0)
	
	combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
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
	
	#initialize ensemble model
	print("initializing ensemble architecture")
	ensemble_model = Ensemble(args.roberta_config, args.random_forest_config, args.logistic_regression_config)

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
