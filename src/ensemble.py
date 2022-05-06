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
from typing import *
from classifier import NNClassifier
from transformers import RobertaForSequenceClassification, RobertaTokenizer

class Ensemble():
	def __init__(self, config, args: argparse.Namespace) -> None:

		super().__init__()
		self.roberta_model = RobertaForSequenceClassification(config)
		self.features_model = nn.LSTM()
		#update to use args and kwargs
		self.classifier = NNClassifier()
		# update classifier to use config?

		#initialize weights

def train():
	#iterate through epochs
		# iterate through data
			# feed regular input to roberta
			# feed features to LSTM
			# concatenate outputs
			# feed through classifier
			# update classifier loss
	return

def main():
	#load data
	print("loading training and development data...")
	train_sentences, train_labels = utils.read_adaptation_data(args.train_data_path)
	dev_sentences,  dev_labels = utils.read_adaptation_data(args.dev_data_path)
	
	
	#initialize ensemble model
	print("initializing ensemble architecture")
	ensemble_model = Ensemble(args.roberta_config, args.random_forest_config, args.logistic_regression_config)

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

	#send to train
	return

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
