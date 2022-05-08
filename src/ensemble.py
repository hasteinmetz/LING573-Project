#option 1 
	# create a class with two models, feed final output into a classifier layer
	#https://towardsdatascience.com/ensembling-huggingfacetransformer-models-f21c260dbb09
#option 2
	# early fusion with roberta as one of the layers

import json
import utils
import torch
nn = torch.nn
import argparse
from typing import *
from classifier import NNClassifier
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

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
	
	def train_random_forest(self, lexical_features, labels) -> None:
		'''
		arguments:
			- config: grid of parameter options to configure rf model with 
			- lexical_features: array of lexical features, one row per sample in data
			- labels: corresponding classification label for each sample
		returns:
			sklearn.ensemble.RandomForestClassifier
		initializes a hyperparameter tuning scheme based off of parameter options provided by config file
		and finds the optimal set of parameters for the best-performing random forest classifier.

		uses cross-fold validation while training.
		'''
		rf = RandomForestClassifier()
		hyperparam_tuner = RandomizedSearchCV(estimator=rf, param_distributions=self.rf_config["param_grid"], n_iter=self.rf_config["n_iter"], \
			cv=self.rf_config["cv"], verbose=self.rf_config["verbose"], random_state=self.rf_config["random_state"], n_jobs=self.rf_config["n_jobs"])
		hyperparam_tuner.fit(lexical_features, labels)
		self.random_forest = hyperparam_tuner.best_estimator_


def train_ensemble(ensemble: Ensemble, train_lex_feat, train_labels, train_data: BatchEncoding) -> None:
	#train random forest
	ensemble.train_random_forest(train_lex_feat, train_labels)

	#get roberta embeddings
	

	#combine rf output and roberta embeddings and feed to logisitical regression model

	#output training performance

def main(args: argparse.Namespace) -> None:
	#load data
	#todo: split data with holdout group
	train_sentences, train_labels = utils.read_data_from_file(args.train_sentences)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_sentences)
	
	#initialize ensemble model
	ensemble_model = Ensemble(args.roberta_config, args.random_forest_config, args.logistical_regression_config)

	#get features

	#get tokenized input
	train_encodings = ensemble_model.roberta_tokenizer.tokenize(train_sentences, return_tensors='pt', padding=True)
	dev_encodings = ensemble_model.roberta_tokenizer.tokenize(dev_sentences, return_tensors='pt', padding=True)

	#send to train
	train_ensemble(ensemble_model, lexical_features, labels, train_encodings)

	#run whole ensemble on dev data 

	#output results

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--roberta_config", help="configuration settings for roberta model")
	parser.add_argument("--random_forest_config", help="configuration settings for random forest classifier")
	parser.add_argument("--logistical_regression_config", help="configuration settings for logistical regression classifier")
	parser.add_argument('--train_data_path', help="path to input training data file")
	parser.add_argument('--dev_data_path', help="path to input dev data file")
	parser.add_argument('--output_file', help="path to output data file")
	parser.add_argument('--save_file', help="path to save the pretrained model", default='None', type=str)
	args = parser.parse_args()

	main(args)
