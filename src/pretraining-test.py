#!/usr/bin/env python

import utils
import torch
import argparse
import sys
import numpy as np
import pandas as pd
from typing import *
from featurizer import featurize, DTFIDF
from torch.optim import AdamW
from transformers import RobertaModel as RoBERTa
from transformers import RobertaTokenizer
from pretraining import EnsembleModel, train_model, evaluate_model, RoBERTaClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import json

nn = torch.nn

def main(args: argparse.Namespace) -> None:
	# check if cuda is avaiable
	DEVICE = "cpu"
	if torch.cuda.is_available():
		DEVICE = "cuda"
		torch.device(DEVICE)
		print(f"Using {DEVICE} device")
		print(f"Using the GPU:{torch.cuda.get_device_name(0)}", file = sys.stderr)
	else:
		torch.device(DEVICE)
		print(f"Using {DEVICE} device")
		print(f"Using {DEVICE} device", file = sys.stderr)

	#load data
	print("Loading training and development data...")
	train_sentences, train_labels = utils.read_data_from_file(args.train_data_path, index=args.index)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_data_path, index=args.index)
	pt_sentences, pt_labels = utils.read_data_from_file(args.train_data_path, index=2)
	dev_pt_sentences, dev_pt_labels = utils.read_data_from_file(args.dev_data_path, index=2)

	if args.debug == 1:
		print(f"NOTE: Running in debug mode", file=sys.stderr)
		train_sentences, train_labels = shuffle(train_sentences, train_labels, random_state = 0)
		dev_sentences, dev_labels = shuffle(dev_sentences, dev_labels, random_state = 0)
		pt_sentences, pt_labels = shuffle(pt_sentences, pt_labels, random_state = 0)
		dev_pt_sentences, dev_pt_labels = shuffle(dev_pt_sentences, dev_pt_labels, random_state = 0)
		train_sentences, train_labels = train_sentences[0:100], train_labels[0:100]
		dev_sentences, dev_labels = dev_sentences[0:10], dev_labels[0:10]
		pt_sentences, pt_labels = pt_sentences[0:50], pt_labels[0:50]
		dev_pt_sentences, dev_pt_labels = dev_pt_sentences[0:50], dev_pt_labels[0:50]
	
	# initialize tf-idf vectorizer
	tfidf = DTFIDF(train_sentences, train_labels)

	# get hurtlex dictionary
	hurtlex_dict, hurtlex_feat_list = utils.read_from_tsv(args.hurtlex_path)

	# load PCA and extract principle components from the training data
	pca = PCA()
	training_feature_matrix = featurize(train_sentences, train_labels, hurtlex_dict, hurtlex_feat_list, tfidf)
	pca.fit(training_feature_matrix)
	print(f"Fitted PCA. Previously there were {training_feature_matrix.shape[1]} " + 
	 	f"features. Now there are {pca.n_components_} features.", file=sys.stderr)

	# reduce the parameters of the featurize function
	FEATURIZER = lambda x: featurize(x, train_labels, hurtlex_dict, hurtlex_feat_list, tfidf)

	# get input size
	input_size = FEATURIZER(train_sentences[0:1]).shape[1]
	
	# LOAD CONFIGURATION
	config_file = f'src/configs/ensemble-{args.job}.json'
	with open(config_file, 'r') as f1:
		configs = f1.read()
		train_config = json.loads(configs)

	# load important parts of the model
	print("Initializing tokenizer...\n")
	TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")

	# load model is already available
	if args.reevaluate != 1:
		print("Pre-training model...\n")

		# initialize the model thingies
		ENSEMBLE = EnsembleModel(input_size, train_config['hidden_size'], train_config['output_size'])
		OPTIMIZER = AdamW
		LOSS_RE = nn.MSELoss()
		LOSS_CL = nn.BCEWithLogitsLoss()
	
		# pretrain the model on the regression task
		train_model(
			model=ENSEMBLE, 
			sentences=pt_sentences,  
			labels=pt_labels,
			test_sents=dev_pt_sentences, 
			test_labels=dev_pt_labels,
			tokenizer=TOKENIZER, 
			featurizer=FEATURIZER,
			optimizer=OPTIMIZER,
			loss_fn=LOSS_RE, 
			device=DEVICE, 
			measures=['mse'],
			epochs=train_config['epochs'], 
			batch_size=train_config['batch_size'], 
			lr=train_config['lr'],
			regression = 'linear'
		)

		# finetune the model on the regression task
		train_model(
			model=ENSEMBLE, 
			sentences=train_sentences, 
			labels=train_labels,
			test_sents=dev_sentences, 
			test_labels=dev_labels,
			tokenizer=TOKENIZER, 
			optimizer=OPTIMIZER,
			featurizer=FEATURIZER,
			loss_fn=LOSS_CL, 
			device=DEVICE, 
			measures = ['f1', 'accuracy'],
			save_path = f"{args.model_save_path}/{args.job}",
			epochs=train_config['epochs'], 
			batch_size=train_config['batch_size'], 
			lr=train_config['lr'],
			regression = 'linear'
		)
	else:
		try:
			print("Loading existing model...\n")
			# TODO: Change the directory organization
			model = torch.load(f'src/models/with-pretraining/{args.job}/')
		except ValueError:
			print("No existing model found. Rerun without --reevaluate")

	# evaluate the model on test data
	print("Evaluating models...")
	preds = evaluate_model(
		model=ENSEMBLE, 
		sentences=dev_sentences, 
		labels=dev_labels, 
		batch_size=train_config['batch_size'],
		tokenizer=TOKENIZER, 
		featurizer=FEATURIZER,
		device=DEVICE, 
		outfile=sys.stdout
	)

	# write results to output file
	dev_out_d = {'sentence': dev_sentences, 'predicted': preds, 'correct_label': dev_labels}
	dev_out = pd.DataFrame(dev_out_d)
	output_file = f'{args.output_path}/{args.job}/pretrain-output.csv'
	dev_out.to_csv(output_file, index=False, encoding='utf-8')

	# filter the data so that only negative examples are there
	data_filtered = dev_out.loc[~(dev_out['predicted'] == dev_out['correct_label'])]
	error_file = f'{args.error_path}-{args.job}.csv'
	data_filtered.to_csv(error_file, index=False, encoding='utf-8')

	print(f"Done! Exited normally :)")

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', help="path to input training data file")
	parser.add_argument('--dev_data_path', help="path to input dev data file")
	parser.add_argument('--hurtlex_path', help="path to hurtlex dictionary")
	parser.add_argument('--job', help="to help name files when running batches", default='test', type=str)
	parser.add_argument('--output_path', help="path to output data file")
	parser.add_argument('--model_save_path', help="path to save models")
	parser.add_argument('--error_path', help="path to save error analysis")
	parser.add_argument('--debug', help="debug the ensemble with small dataset", type=int)
	parser.add_argument('--index', help="column to select from data", type=int)
	parser.add_argument('--reevaluate', help="use 1 to reload an existing model if already completed", type=int, default=0)
	args = parser.parse_args()

	main(args)
