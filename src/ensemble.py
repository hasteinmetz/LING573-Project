#option 1 
	# create a class with two models, feed final output into a classifier layer
	#https://towardsdatascience.com/ensembling-huggingfacetransformer-models-f21c260dbb09
#option 2
	# early fusion with roberta as one of the layers

import json
import common_utils
import torch
import argparse
import numpy as np
from typing import *
from featurizer import featurize
from classifier import NNClassifier
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

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
		self.rf_config = common_utils.load_json_config(forest_config_path)
		self.random_forest = None

		logreg_config = common_utils.load_json_config(logreg_config_path)
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


def train_ensemble(ensemble: Ensemble, train_lex_feat: np.ndarray, train_labels: np.ndarray, roberta_input: BatchEncoding, device: str) -> None:
	#train random forest
	ensemble.train_random_forest(train_lex_feat, train_labels)
	rf_class_prob = ensemble.random_forest.predict_proba(train_lex_feat)

	#get roberta embeddings
	ensemble.roberta_model.eval()
	ensemble.roberta_model.to(device)
	roberta_class_prob = None
	with torch.no_grad():
		outputs = ensemble.roberta_model(**roberta_input)
		logits = outputs.logits
		roberta_class_prob = logits.clone().detach().to('cpu').numpy()

	#combine rf output and roberta embeddings and feed to logisitical regression model
	combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
	ensemble.classifier.fit(combined_class_prob, train_labels)

	#output training performance
	training_accuracy = ensemble.classifier.score(combined_class_prob, train_labels)
	print("training accuracy: {}".format(training_accuracy))

def predict(ensemble: Ensemble, dev_lex_feat: np.ndarray, dev_labels: np.ndarray, roberta_input: BatchEncoding, device: str) -> np.ndarray:
	rf_class_prob = ensemble.random_forest.predict_proba(dev_lex_feat)

	ensemble.roberta_model.eval()
	ensemble.roberta_model.to(device)
	roberta_class_prob = None
	with torch.no_grad():
		outputs = ensemble.roberta_model(**roberta_input)
		logits = outputs.logits
		roberta_class_prob = logits.clone().detach().to('cpu').numpy()
	combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
	predicted_labels = ensemble.classifier.predict(combined_class_prob)
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
	train_sentences, train_labels = common_utils.read_data_from_file(args.train_sentences)
	dev_sentences, dev_labels = common_utils.read_data_from_file(args.dev_sentences)
	
	#initialize ensemble model
	ensemble_model = Ensemble(args.roberta_config, args.random_forest_config, args.logistical_regression_config)

	#get features
	train_feature_vector = featurize(train_sentences, train_labels)
	dev_feature_vector = featurize(dev_sentences, dev_labels)

	#get tokenized input
	train_encodings = ensemble_model.roberta_tokenizer.tokenize(train_sentences, return_tensors='pt', padding=True)
	dev_encodings = ensemble_model.roberta_tokenizer.tokenize(dev_sentences, return_tensors='pt', padding=True)

	#send to train
	train_ensemble(ensemble_model, train_feature_vector, train_labels, train_encodings, device)

	#run whole ensemble on dev data 
	dev_predicted_labels = predict(ensemble_model, dev_feature_vector, dev_labels, dev_encodings, device)

	#output results
	dev_out_d = {'sentence': dev_sentences, 'predicted': dev_predicted_labels, 'correct_label': dev_labels}
	dev_out = pd.DataFrame(dev_out_d)
	dev_out.to_csv(args.output_file, index=False, encoding='utf-8')

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