import json
import utils
import torch
import argparse
import numpy as np
import pandas as pd
from typing import *
from featurizer import featurize
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
		print(accuracy_df)


def train_ensemble(ensemble: Ensemble, train_lex_feat: np.ndarray, train_labels: np.ndarray, roberta_input: FineTuneDataSet, device: str) -> None:
	#train random forest
	print("\ttraining random forest classifier...")
	ensemble.train_random_forest(train_lex_feat, train_labels)
	rf_class_prob = ensemble.random_forest.predict_proba(train_lex_feat)


	#get roberta embeddings
	ensemble.roberta_model.eval()
	ensemble.roberta_model.to(device)
	roberta_class_prob = None
	eval_dataloader = DataLoader(roberta_input, batch_size=1)
	for batch in eval_dataloader:
		batch["labels"] = batch.pop('label')
		batch = {k: v.to(device) for k, v in batch.items()}
		print(batch.shape())

		with torch.no_grad():
			outputs = ensemble.roberta_model(**batch)
		logits = outputs.logits
		roberta_class_prob = logits.clone().detach().to('cpu').numpy()

	#combine rf output and roberta embeddings and feed to logisitical regression model
	combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
	print("\ttraining logistic regression classifier")
	ensemble.classifier.fit(combined_class_prob, train_labels)

	#output training performance
	training_accuracy = ensemble.classifier.score(combined_class_prob, train_labels)
	print("training accuracy: {}".format(training_accuracy))

def predict(ensemble: Ensemble, dev_lex_feat: np.ndarray, dev_labels: np.ndarray, roberta_input: FineTuneDataSet, device: str) -> np.ndarray:
	rf_class_prob = ensemble.random_forest.predict_proba(dev_lex_feat)

	ensemble.roberta_model.eval()
	ensemble.roberta_model.to(device)
	roberta_class_prob = None
	# convert dataset to a pytorch format and batch the data
	eval_dataloader = DataLoader(roberta_input, batch_size=1)

	# iterate through batches to get outputs
	for batch in eval_dataloader:
		# assign each element of the batch to the device
		batch["labels"] = batch.pop('label')
		batch = {k: v.to(device) for k, v in batch.items()}
		print(batch.shape())
	
		# get batched results
		with torch.no_grad():
			outputs = ensemble.roberta_model(**batch)
		logits = outputs.logits
		roberta_class_prob = logits.clone().detach().to('cpu').numpy()

	combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
	predicted_labels = ensemble.classifier.predict(combined_class_prob)
	predicted_accuracy = ensemble.classifier.score(combined_class_prob, dev_labels)
	print("\tdev accuracy: {}".format(predicted_accuracy))
	return predicted_labels

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
	print("featurizing training and dev data...")
	train_feature_vector = featurize(train_sentences, train_labels)
	dev_feature_vector = featurize(dev_sentences, dev_labels)

	#get tokenized input
	print("preparing input for roberta model...")
	train_data = FineTuneDataSet(train_sentences, train_labels)
	train_data.tokenize_data(ensemble_model.roberta_tokenizer)

	dev_data = FineTuneDataSet(dev_sentences, dev_labels)
	dev_data.tokenize_data(ensemble_model.roberta_tokenizer)

	#send to train
	print("training ensemble model...")
	train_ensemble(ensemble_model, train_feature_vector, train_labels, train_data, device)

	#run whole ensemble on dev data 
	print("predicting dev labels...")
	dev_predicted_labels = predict(ensemble_model, dev_feature_vector, dev_labels, dev_data, device)

	#output results
	print("outputting dev classification output...")
	dev_out_d = {'sentence': dev_sentences, 'predicted': dev_predicted_labels, 'correct_label': dev_labels}
	dev_out = pd.DataFrame(dev_out_d)
	dev_out.to_csv(args.output_file, index=False, encoding='utf-8')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--roberta_config", help="configuration settings for roberta model")
	parser.add_argument("--random_forest_config", help="configuration settings for random forest classifier")
	parser.add_argument("--logistic_regression_config", help="configuration settings for logistic regression classifier")
	parser.add_argument('--train_data_path', help="path to input training data file")
	parser.add_argument('--dev_data_path', help="path to input dev data file")
	parser.add_argument('--output_file', help="path to output data file")
	args = parser.parse_args()

	main(args)
