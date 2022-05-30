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
from featurizer import featurize, DTFIDF
from torch.utils.data import DataLoader
from feature_selection import k_perc_best_f, prune_test
from pytorch_utils import FineTuneDataSet, make_labels_binary
from torch.optim import AdamW, SGD, Adagrad
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import json

nn = torch.nn

torch.manual_seed(770613)

class RoBERTa(nn.Module):
	'''A wrapper around the RoBERTa model with a defined forward function'''
	def __init__(self, dropout_roberta: float = 0.1):
		super(RoBERTa, self).__init__()
		self.roberta = RobertaForSequenceClassification.from_pretrained(
			pretrained_model_name_or_path = 'roberta-base', 
			hidden_dropout_prob = dropout_roberta, 
			attention_probs_dropout_prob = dropout_roberta, 
			problem_type = "single_label_classification"
		)

	def forward(self, data: dict, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		roberta_out = self.roberta(**inputs)
		return roberta_out

class FeatureClassifier(nn.Module):
	'''Simple feed forward neural network to classify a word's lexical features'''
	def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float):
		'''MLP composed of 4 hidden layers with dropout'''
		super(FeatureClassifier, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout), 
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, num_classes)
		)

	def forward(self, features: torch.tensor):		
		logits = self.mlp(features)
		return logits

class LogisticRegression(nn.Module):
	'''Using PyTorch's loss functions and layers to model Logistic Regression'''
	def __init__(self, input_dim: int):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_dim, 1)

	def forward(self, input_logits: torch.tensor):
		linear = self.linear(input_logits)
		return linear

def train_ensemble(
		Transformer: RoBERTa, FClassifier: FeatureClassifier, LogRegressor: LogisticRegression, tokenizer: RobertaTokenizer,
		sentences: List[str], labels: np.ndarray, featurizer: Callable,
		test_sentences: List[str], test_labels: np.ndarray,
		epochs: int, batch_size: int, 
		lr_transformer: float, lr_classifier: float, lr_regressor: float,
		kfolds: int,
		optimizer_transformer: torch.optim, optimizer_classifier: torch.optim, optimizer_regression: torch.optim,
		loss_fn: Callable, device: str, save_path: str, dim_spec: str
	):
	'''Train the ensemble on the training data'''

	# send the models to the device
	Transformer.to(device)
	FClassifier.to(device)
	LogRegressor.to(device)

	# initialize the optimizers for the Transformer and FClassifier
	optim_tr = optimizer_transformer(Transformer.parameters(), lr=lr_transformer, weight_decay=1e-5)
	optim_cl = optimizer_classifier(FClassifier.parameters(), lr=lr_classifier, weight_decay=1e-5)
	optim_log = optimizer_regression(LogRegressor.parameters(), lr=lr_regressor, weight_decay=1e-5)

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)
	
	# set up regression loss function
	loss_logistic = nn.BCEWithLogitsLoss()

	# prepare the kfolds cross validator
	# k-folds cross-validation trains the Transformer and FClassifier on parts of the
	# test data and then LogRegressor on the remaining dataset (outputs provided by the other two models)
	kfolder = StratifiedKFold(n_splits=kfolds)
	kfolder.get_n_splits(shuffled_sentences, shuffled_labels)

	for epoch in range(epochs):

		for fold, (train_i, test_i) in enumerate(kfolder.split(shuffled_sentences, shuffled_labels)):

			X = np.array(shuffled_sentences)
			y = shuffled_labels

			# split into models and metamodel training data
			X_models, y_models = X[train_i].tolist(), y[train_i]
			X_meta, y_meta = X[test_i].tolist(), y[test_i]

			# make the data a Dataset and put it in a DataLoader to batch it
			dataset = FineTuneDataSet(X_models, y_models, verbose = 'verbose')
			dataset.tokenize_data(tokenizer)
			dataloader = DataLoader(dataset, batch_size=batch_size)

			# TRAIN THE TRANSFORMER AND THE FCLASSIFIER
			Transformer.train()
			FClassifier.train()
			LogRegressor.train()

			for i, (batch, X) in enumerate(dataloader):

				# make the labels of type torch.long
				batch['labels'] = batch['labels'].type(torch.long).to(device)

				# get the labels as a separate variable
				y = batch['labels']
				y.to(device)

				# TRANSFORMER TRAINING

				# set transformer optimizer to 0-grad
				optim_tr.zero_grad()

				# NOTE: RoBERTa needs labels to be a tensor torch.long or torch.int
				# in order to compute the loss for a single class
				transformer_outputs = Transformer(batch, device)
				
				loss_tr = transformer_outputs.loss
				loss_tr.backward()
				optim_tr.step()

				# FCLASSIFIER TRAINING

				# set classifier optimizer to 0-grad
				optim_cl.zero_grad()

				# initialize the feature tensor
				features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)

				classifier_outputs = FClassifier(features_tensor)

				loss_cl = loss_fn(classifier_outputs, y)
				loss_cl.backward()
				optim_cl.step()

				if (i + 1) % 10 == 0:
			
					# output metrics to standard output
					print(
						f'\t(epoch {epoch+1}, fold {fold+1}, samples {(i+1)*batch_size}) ' +
						f'FClassifier Loss: {loss_cl.item()} Transformer Loss: {loss_tr.item()}', 
						file = sys.stderr
					)

			# make the data a Dataset and put it in a DataLoader to batch it
			dataset = FineTuneDataSet(X_meta, y_meta, verbose = 'verbose')
			dataset.tokenize_data(tokenizer)
			dataloader = DataLoader(dataset, batch_size=batch_size)

			# TRAIN THE LOGISTIC REGRESSOR
			Transformer.eval()
			FClassifier.eval()

			for i, (batch, X) in enumerate(dataloader):

				optim_log.zero_grad()

				# GET LOGITS

				with torch.no_grad():
					# make the labels of type torch.long
					batch['labels'] = batch['labels'].type(torch.long).to(device)

					# transformer logits
					transformer_logits = Transformer(batch, device).logits
					
					# classifier logits
					features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)
					feature_logits = FClassifier(features_tensor)

				# TRAIN REGRESSOR
				all_logits = torch.cat((transformer_logits, feature_logits), axis=1)
				output = LogRegressor(all_logits)
				
				# reshape true labels
				y = torch.reshape(batch['labels'], (batch['labels'].size()[0], 1)).float()

				# get loss and do backpropagation
				loss_lg = loss_logistic(output, y)
				loss_lg.backward()
				optim_log.step()

				if (i + 1) % 20 == 0:
			
					# output metrics to standard output
					probability = torch.sigmoid(output)
					correct = (torch.round(probability) == y).type(torch.float).sum().item() 
					total = output.shape[0]
					accuracy = correct/total
					print(
						f'(epoch {epoch+1}, fold {fold+1}, samples {(i+1)*batch_size}) ' +
						f'Regression Accuracy: {accuracy}, Loss: {loss_lg.item()}',
						file = sys.stderr
					)
				
			# Get the accuracy of the fold on the test data
			print(f"Fold {fold} accuracies:", file = sys.stderr)
			result, preds1, preds2, preds3 = evaluate_ensemble(Transformer, FClassifier, LogRegressor, tokenizer, test_sentences, test_labels, featurizer, batch_size, device, sys.stderr)
					
	# SAVE MODEL
	try:
		print(f"Saving models to /src/models/{args.job}/...", file=sys.stderr)
		torch.save(Transformer, save_path + f'/{args.job}/roberta-{dim_spec}.pt')
		torch.save(FClassifier, save_path + f'/{args.job}/featurizer-{dim_spec}.pt')
		torch.save(LogRegressor, save_path + f'/{args.job}/regression-{dim_spec}.pt')
	except Exception("Could not save model..."):
		print(f"(Saving error) Couldn't save model to {save_path}/{args.job}/...", file=sys.stderr)


def evaluate_ensemble(
		Transformer: RoBERTa, FClassifier: FeatureClassifier, LogRegressor: LogisticRegression, tokenizer: RobertaTokenizer,
		sentences: List[str], labels: np.ndarray, featurizer: Callable, batch_size: int, device: str, file: object = sys.stdout
	):
	'''Evaluate the Ensemble'''
	
	# send the models to the device
	Transformer.to(device)
	FClassifier.to(device)

	# turn the models into eval mode
	Transformer.eval()
	FClassifier.eval()
	LogRegressor.eval()

	metrics, tr_metrics, cl_metrics = [], [], []
	for metric in ['f1', 'accuracy']:
		metrics.append(load_metric(metric))
		tr_metrics.append(load_metric(metric))
		cl_metrics.append(load_metric(metric))

	# make the data a Dataset and put it in a DataLoader to batch it
	dataset = FineTuneDataSet(sentences, labels, verbose = 'verbose')
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	# intialize a list to store predictions
	predictions, roberta_preds, feature_preds = [], [], []

	for batch, X in dataloader:

		# make the labels of type torch.long
		batch['labels'] = batch['labels'].type(torch.long).to(device)
		y = batch['labels']

		# GET LOGITS

		with torch.no_grad():
			# transformer logits
			transformer_logits = Transformer(batch, device).logits
			
			# classifier logits
			features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)
			feature_logits = FClassifier(features_tensor)
			
		# OUTPUT EACH CLASSIFIER'S RESULT INDIVIDUALLY
		tr_argmax = torch.argmax(transformer_logits, dim = -1)
		cl_argmax = torch.argmax(feature_logits, dim = -1)
		for m1 in tr_metrics:
			m1.add_batch(predictions=tr_argmax, references=y)
		for m2 in cl_metrics:
			m2.add_batch(predictions=cl_argmax, references=y)

		# EVALUATE REGRESSOR
		all_logits = torch.cat((transformer_logits, feature_logits), axis=1)
		y_hats = LogRegressor(all_logits)

		# add batch to output
		probability = torch.sigmoid(y_hats)
		as_list = torch.round(probability).to('cpu').tolist()
		roberta_preds.extend(tr_argmax.to('cpu').tolist())
		feature_preds.extend(cl_argmax.to('cpu').tolist())
		predictions.extend(as_list)

		# add batched results to metrics
		for m3 in metrics:
			m3.add_batch(predictions=torch.round(probability), references=y)

	# output metrics to standard output
	val_en, val_tr, val_cl = f"", f"", f"" # empty strings
	for m1, m2, m3 in zip(metrics, tr_metrics, cl_metrics):
		val1, val2, val3 = m1.compute(), m2.compute(), m3.compute()
		val_en += f"{m1.name}: {val1}\n"
		val_tr += f"Subscore: Transformer\t{m2.name}: {val2}\n"
		val_cl += f"Subscore: Featurizer:\t{m3.name}: {val3}\n"
		# output metrics to standard output
	result = "\n".join([val_en, val_tr, val_cl])
	print(result.split("\n")[0:2], file=file)
	print(result, file=sys.stderr)
	return result, predictions, roberta_preds, feature_preds


def main(args: argparse.Namespace) -> None:
	# check if cuda is avaiable
	DEVICE = "cpu"
	if torch.cuda.is_available():
		DEVICE = "cuda"
		torch.device(DEVICE)
		print(f"Using {DEVICE} device", file=sys.stderr)
		print(f"Using the GPU:{torch.cuda.get_device_name(0)}", file = sys.stderr)
	else:
		torch.device(DEVICE)
		print(f"Using {DEVICE} device", file=sys.stderr)
		print(f"Using {DEVICE} device", file = sys.stderr)

	#load data
	print("Loading training and development data...", file=sys.stderr)
	dev_path = args.test_data_path + "dev.csv"
	train_sentences, train_labels = utils.read_data_from_file(args.train_data_path, index=args.index)
	dev_sentences, dev_labels = utils.read_data_from_file(dev_path, index=args.index)

	if args.debug == 1:
		print(f"NOTE: Running in debug mode", file=sys.stderr)
		train_sentences, train_labels = shuffle(train_sentences, train_labels, random_state = 0)
		dev_sentences, dev_labels = shuffle(dev_sentences, dev_labels, random_state = 0)
		train_sentences, train_labels = train_sentences[0:100], train_labels[0:100]
		dev_sentences, dev_labels = dev_sentences[0:10], dev_labels[0:10]

	# initialize tf-idf vectorizer
	tfidf = DTFIDF(train_sentences, train_labels)

	# get hurtlex dictionary
	hurtlex_dict, hurtlex_feat_list = utils.read_from_tsv(args.hurtlex_path)

	print("reducing feature dimensions...", file=sys.stderr)
	if args.dim_reduc_method == 'pca':
		train_feature_vector = featurize(train_sentences, hurtlex_dict, hurtlex_feat_list, tfidf)
		train_pca = PCA(.95)
		train_pca.fit(train_feature_vector)
		print("\tnum components: {}".format(train_pca.n_components))
		FEATURIZER = lambda x: train_pca.transform(featurize(x, hurtlex_dict, hurtlex_feat_list, tfidf))
	else:
		train_feat_vector = featurize(train_sentences, hurtlex_dict, hurtlex_feat_list, tfidf)
		train_feature_vector, feat_indices = k_perc_best_f(train_feat_vector, train_labels, 70)
		# use the features inside the model using a featurize function
		FEATURIZER = lambda x: prune_test(featurize(x, hurtlex_dict, hurtlex_feat_list, tfidf), feat_indices)

	# get input size
	input_size = FEATURIZER(train_sentences[0:1]).shape[1]

	# LOAD CONFIGURATION
	config_file = f'src/configs/nn_kfolds_{args.job}.json'
	with open(config_file, 'r') as f1:
		configs = f1.read()
		train_config = json.loads(configs)
		print(f"Config: {train_config}")

	# initialize ensemble model
	print("Initializing ensemble architecture...\n", file=sys.stderr)

	# optimizers and loss functions
	OPTIMIZER_TRANSFORMER = AdamW
	OPTIMIZER_CLASSIFIER = Adagrad
	OPTIMIZER_REGRESSOR = SGD
	LOSS = nn.CrossEntropyLoss()

	# roberta
	dropout_roberta = train_config.pop('dropout_roberta')
	TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
	ROBERTA = RoBERTa(dropout_roberta=dropout_roberta)

	# MLP and regressor
	hidden_layers = train_config.pop('hidden_size')
	dropout = train_config.pop('dropout_mlp')
	FEATURECLASSIFIER = FeatureClassifier(input_size, hidden_layers, 2, dropout)
	LOGREGRESSION = LogisticRegression(4)

	# train the model
	train_ensemble(
		Transformer=ROBERTA, 
		FClassifier=FEATURECLASSIFIER, 
		LogRegressor=LOGREGRESSION, 
		tokenizer=TOKENIZER,
		sentences=train_sentences, 
		labels=train_labels, 
		featurizer=FEATURIZER, 
		test_sentences=dev_sentences, 
		test_labels=dev_labels,
		optimizer_transformer=OPTIMIZER_TRANSFORMER, 
		optimizer_classifier=OPTIMIZER_CLASSIFIER, 
		optimizer_regression=OPTIMIZER_REGRESSOR,
		loss_fn=LOSS, 
		device=DEVICE, 
		save_path=args.model_save_path,
		dim_spec=args.dim_reduc_method,
		**train_config
	)

	# try:
	# 	print("Loading the models...", file=sys.stderr)
	# 	ROBERTA = torch.load(f'{args.model_save_path}/{args.job}/roberta-{args.dim_reduc_method}.pt')
	# 	FEATURECLASSIFIER = torch.load(f'{args.model_save_path}/{args.job}/featurizer-{args.dim_reduc_method}.pt')
	# 	LOGREGRESSION = torch.load(f'{args.model_save_path}/{args.job}/regression-{args.dim_reduc_method}.pt')
	# except Exception("Could not load models..."):
	# 	print("Couldn't load the models... Using the most recently trained model.")

	# evaluate the model on test data
	print("Evaluating models...", file=sys.stderr)
	with open(f"{args.output_path}/{args.job}/devtest/D4_scores.out", 'w') as outfile:
		outfile.write("########################\n\tD4 DEV SCORES\n########################\n")
		print("########################\n\tD4 DEV SCORES\n########################\n")
		result, preds, robs, feats = evaluate_ensemble(
			ROBERTA, FEATURECLASSIFIER, LOGREGRESSION, TOKENIZER,
			dev_sentences, dev_labels, FEATURIZER, train_config['batch_size'], DEVICE, file=outfile
		)
		print(result)

	# write results to output file
	test_out_d = {'sentence': dev_sentences, 'predicted': preds, 'transformer': robs, 'featurizer': feats, 'correct_label': dev_labels}
	test_out = pd.DataFrame(test_out_d)
	output_file = f'{args.output_path}/{args.job}/devtest/D4.csv'
	test_out.to_csv(output_file, index=False, encoding='utf-8')

	# filter the data so that only negative examples are there
	data_filtered = test_out.loc[~(test_out['predicted'] == test_out['correct_label'])]
	error_file = f'{args.error_path}_{args.job}_devtest.csv'
	data_filtered.to_csv(error_file, index=False, encoding='utf-8')


	if args.test == 'eval':

		print("Loading test data...", file=sys.stderr)
		test_path = args.test_data_path + "test.csv"
		test_sentences, test_labels = utils.read_data_from_file(test_path, index=args.index)

		# evaluate the model on test data
		with open(f"{args.output_path}/{args.job}/evaltest/D4_scores.out", 'w') as outfile:
			print("########################\n\tD4 TEST SCORES\n########################")
			print("########################\n\tD4 TEST SCORES\n########################", file=outfile)
			result, preds, robs, feats = evaluate_ensemble(
				ROBERTA, FEATURECLASSIFIER, LOGREGRESSION, TOKENIZER,
				dev_sentences, dev_labels, FEATURIZER, train_config['batch_size'], DEVICE, file=outfile
			)
			print(result)

		# write results to output file
		test_out_d = {'sentence': test_sentences, 'predicted': preds, 'transformer': robs, 'featurizer': feats, 'correct_label': test_labels}
		test_out = pd.DataFrame(test_out_d)
		output_file = f'{args.output_path}/{args.job}/evaltest/D4_scores.csv'
		test_out.to_csv(output_file, index=False, encoding='utf-8')

		# filter the data so that only negative examples are there
		data_filtered = test_out.loc[~(test_out['predicted'] == test_out['correct_label'])]
		error_file = f'{args.error_path}_{args.job}_evaltest.csv'
		data_filtered.to_csv(error_file, index=False, encoding='utf-8')

		print(f"Done! Exited normally :)", file=sys.stderr)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', help="path to input training data file")
	parser.add_argument('--test_data_path', help="path to input dev data file")
	parser.add_argument('--hurtlex_path', help="path to hurtlex dictionary")
	parser.add_argument('--output_path', help="path to output data file")
	parser.add_argument('--dim_reduc_method', help="method used to reduce the dimensionality of feature vectors", default = 'pca')
	parser.add_argument('--model_save_path', help="path to save models")
	parser.add_argument('--error_path', help="path to save error analysis")
	parser.add_argument('--job', help="job being done (humor or controversy) to help with folder organization")
	parser.add_argument('--debug', help="debug the ensemble with small dataset", type=int)
	parser.add_argument('--index', help="column to select from data", type=int)
	parser.add_argument('--test', help="whether to run the model on the test set", type=str, default='dev')
	args = parser.parse_args()

	main(args)
