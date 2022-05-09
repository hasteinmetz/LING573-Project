#option 1 
	# create a class with two models, feed final output into a classifier layer
	#https://towardsdatascience.com/ensembling-huggingfacetransformer-models-f21c260dbb09
#option 2
	# early fusion with roberta as one of the layers

''' References:
	- https://www.kaggle.com/code/ynouri/random-forest-k-fold-cross-validation/notebook
'''

import utils
import torch
import argparse
import numpy as np
import pandas as pd
from typing import *
from feature_selection import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from finetune_dataset import FineTuneDataSet
from featurizer import featurize, get_all_features
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

nn = torch.nn
import argparse
import featurizer
from typing import *
from featurizer import featurize
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer, get_scheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from fine_tune import *

nn = torch.nn

class Ensemble():
	def __init__(self, forest_config_path: str, logreg_config_path: str) -> None:
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
		self.roberta_model = RobertaModelWrapper(32, 0.00005)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		self.rf_config = utils.load_json_config(forest_config_path)
		self.random_forest = None
		if 'src/model' in self.roberta_config._name_or_path:
			self.pretrained = 'fine-tuned'
		else:
			self.pretrained = 'roberta-base'

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
		#hyperparam_tuner = RandomizedSearchCV(estimator=rf, param_distributions=self.rf_config["param_grid"], n_iter=self.rf_config["n_iter"], \
		#	cv=self.rf_config["cv"], verbose=self.rf_config["verbose"], random_state=self.rf_config["random_state"], n_jobs=self.rf_config["n_jobs"])
		hyperparam_tuner.fit(lexical_features, labels)
		self.random_forest = hyperparam_tuner.best_estimator_


def get_ensemble_inputs(data: FineTuneDataSet, model: Ensemble) -> tuple[np.ndarray, np.ndarray]:
	'''Helper function that takes input sentences and return ndarrays of the sentence's feature vector and roberta encodings'''
	feature_vector = featurize(data.sentences, train_labels)
	data.tokenize_data(model.roberta_model.tokenizer)
	return feature_vector, DataLoader(data)


def train_ensemble(ensemble: Ensemble, train_sentences: List[str], train_labels: List[str], device: str) -> None:

	# split the training data into 7-folds
	print("\tsplitting data in k-folds to cross-validate...")
	kfolds = StratifiedKFold(n_splits=7)
	kfolds.get_n_splits(train_sentences, train_labels)
	
	# put the model into train mode and send to GPU
	ensemble.roberta_model.train()
	ensemble.roberta_model.to(device)

	for train_index, test_index in kfolds.split(train_sentences, train_labels):

		base_models_train, base_models_labels = train_sentences[train_index], train_labels[train_index]
		meta_model_train, meta_model_labels = train_sentences[test_index], train_labels[test_index]

		base_models_data = FineTuneDataSet(base_models_train, base_models_labels)
		meta_model_data = FineTuneDataSet(meta_model_train, meta_model_labels)

		# get the inputs for the corresponding training data
		train_lex_feat, train_roberta_input = get_ensemble_inputs(base_models_data, ensemble)
		meta_lex_feat, meta_roberta_input = get_ensemble_inputs(meta_model_data, ensemble)

		print(f"\t((train-fold {train_index}, test_fold {test_index}) training random forest classifier...")

		ensemble.train_random_forest(train_lex_feat, base_models_labels)
		rf_class_prob = ensemble.random_forest.predict_proba(meta_lex_feat)

		# train roberta
		ensemble.roberta_model.train(train_roberta_input, ['f1', 'accuracy'], device)

		# get roberta predictions
		roberta_class_prob = ensemble.roberta_model.train(evaluate, ['f1', 'accuracy'], device)

		# combine rf output and roberta embeddings and feed to logisitical regression model
		combined_class_prob = np.concatenate((roberta_class_prob, rf_class_prob), axis=1)
		print("\ttraining logistic regression classifier")
		ensemble.classifier.fit(combined_class_prob, meta_model_labels)

		#output training performance
		training_accuracy = ensemble.classifier.score(combined_class_prob, meta_model_labels)
		print("training accuracy: {}".format(training_accuracy))


def predict(ensemble: Ensemble, dev_sentences: List[str], dev_labels: List[str], device: str) -> np.ndarray:
	# get the feature vectors and embeddings
	dev_lex_feat, dev_roberta_input = get_ensemble_inputs(dev_sentences, dev_labels, ensemble)
	rf_class_prob = ensemble.random_forest.predict_proba(dev_lex_feat)
	roberta_class_prob = ensemble.roberta_model.train(evaluate, ['f1', 'accuracy'], device)
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
	train_sentences, train_labels = utils.read_adaptation_data(args.train_data_path)
	dev_sentences,  dev_labels = utils.read_adaptation_data(args.dev_data_path)
	
	
	#initialize ensemble model
	print("initializing ensemble architecture")
	ensemble_model = Ensemble(args.random_forest_config, args.logistic_regression_config)

	#get features
	print("preparing hurtlex dictionary...")
	hurtlex_dict, hurtlex_feat_list = utils.read_from_tsv(args.hurtlex_path)
	print("featurizing training and dev data...")
	train_feature_vector, dev_feature_vector = [],[]
	if args.dim_reduc_method == 'pca':
		train_feature_vector, dev_feature_vector  = get_all_features(train_sentences, dev_sentences, hurtlex_dict, hurtlex_feat_list)
	else:
		train_feat_vector = featurize(train_sentences, hurtlex_dict, hurtlex_feat_list)
		dev_feat_vector = featurize(dev_sentences, hurtlex_dict, hurtlex_feat_list)
		print("reducing feature dimensions...")
		train_feature_vector, feat_indices = k_perc_best_f(train_feat_vector, train_labels, 70)
		dev_feature_vector = prune_test(dev_feat_vector, feat_indices)

	#get tokenized input
	print("preparing input for roberta model...")

	train_dataset = FineTuneDataSet(train_sentences, train_labels)
	dev_dataset = FineTuneDataSet(dev_sentences, dev_labels)
	train_dataset.tokenize_data(ensemble_model.roberta_tokenizer)
	dev_dataset.tokenize_data(ensemble_model.roberta_tokenizer)

	#get tokenized input
	print("tokenizing inputs for roberta model...")
	train_encodings = ensemble_model.roberta_tokenizer(train_sentences, return_tensors='pt', padding=True)
	dev_encodings = ensemble_model.roberta_tokenizer(dev_sentences, return_tensors='pt', padding=True)

	#send to train
	print("training ensemble model...")
	train_ensemble(ensemble_model, train_sentences, train_labels, device)

	#run whole ensemble on dev data 
	print("predicting dev labels...")
	dev_predicted_labels = predict(ensemble_model, dev_feature_vector, dev_labels, dev_encodings, device)

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
	parser.add_argument("--dim_reduc_method", help="which method to use for reducing feature vector dimensions", choices=['mutual_info','pca'])
	parser.add_argument("--train_data_path", help="path to input training data file")
	parser.add_argument("--dev_data_path", help="path to input dev data file")
	parser.add_argument("--hurtlex_path", help="path to hurtlex lexicon file")
	parser.add_argument("--output_file", help="path to output data file")
	parser.add_argument("--results_file", help="path to which accuracy and f1 score will be written to")
	args = parser.parse_args()

	main(args)
