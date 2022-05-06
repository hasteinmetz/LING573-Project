#option 1 
	# create a class with two models, feed final output into a classifier layer
	#https://towardsdatascience.com/ensembling-huggingfacetransformer-models-f21c260dbb09
#option 2
	# early fusion with roberta as one of the layers

import torch
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

	#get features

	#send to train
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--roberta_config", help="configuration settings for roberta model")
	parser.add_argument("--feature_model_config", help="configuration settings for feature_model")
	parser.add_argument("--roberta_folder", help="path to pretrained roberta model", default=None)
	parser.add_argument("--feature_model_folder", help="path to pretrained feature_model")
	parser.add_argument('--train_sentences', help="path to input training data file")
	parser.add_argument('--dev_sentences', help="path to input dev data file")
	parser.add_argument('--output_file', help="path to output data file")
	parser.add_argument('--save_file', help="path to save the pretrained model", default='None', type=str)
