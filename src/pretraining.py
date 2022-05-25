#!/usr/bin/env python

'''
This file contains classes of models
'''

import utils
import torch
import argparse
import sys
import numpy as np
import pandas as pd
from typing import *
from featurizer import featurize, DTFIDF
from torch.utils.data import DataLoader, Dataset
from custom_pytorch_utils import FineTuneDataSet, make_torch_labels_binary
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification as RobertaSeqCls
from datasets import load_metric
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import json

nn = torch.nn

################################################################################################
###################	MODELS AS NN MODULES
################################################################################################
	
class RoBERTaRegression(nn.Module):
	'''A regression classifier head that uses RoBERTa outputs a single digit
		- Should be trained on nn.MSELoss()
		This is to be used as a template for other classes
	'''
	def __init__(self):
		super(RoBERTaRegression, self).__init__()
		self.roberta = RobertaModel.from_pretrained('roberta-base')
		roberta_hidden_size = self.roberta.config.hidden_size
		self.regression = nn.Linear(roberta_hidden_size, 1)

	def forward(self, data: dict, featurizer: Callable, sentences: List[str], device: str):
		'''
		Forward pass. Note that featurizer and sentences d nothing; 
		it's included for compatibility with train_linear_regression()
		'''
		inputs = {k:v.to(device) for k,v in data.items()}
		# use the pooler outputs for the regression layer (as in the HuggingFace documentation)
		outputs_roberta = self.roberta(**inputs).pooler_output
		pred = self.regression(outputs_roberta)
		return pred


class RoBERTaClassifier(nn.Module):
	'''A wrapper around the RoBERTa model with a defined forward function'''
	def __init__(self):
		super(RoBERTaClassifier, self).__init__()
		self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')

	def forward(self, data: dict, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		roberta_out = self.roberta(**inputs)
		return roberta_out


class FeatureRegression(nn.Module):
	'''Run an MLP with a regression head to classify the features'''
	def __init__(self, input_size: int, dropout: float, hidden_size: int):
		super(FeatureClassifier, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout), # high-ish dropout to avoid overfitting to certain features
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, 1)
		)

	def forward(self, features: torch.tensor):		
		logits = self.mlp(features)
		return logits


class FeatureClassifier(nn.Module):
	'''Simple feed forward neural network to classify a word's lexical features'''
	def __init__(self, input_size: int, dropout: float, hidden_size: int, num_classes: int):
		super(FeatureClassifier, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout), # high-ish dropout to avoid overfitting to certain features
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, num_classes)
		)

	def forward(self, features: torch.tensor):		
		logits = self.mlp(features)
		return logits


class Regression(nn.Module):
	'''Using PyTorch's loss functions and layers to model linear (and logistic) on the inputs
		For logistic regression, you can set the output_fn to torch.sigmoid
		The sort of regression you want to use depends on the loss function
	'''
	def __init__(self, input_dim: int, output_fn = lambda x: x):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_dim, 1)
		self.out_fn = output_fn

	def forward(self, input_logits: torch.tensor):
		linear = self.linear(input_logits)
		return self.out_fn(linear)


class EnsembleModel(nn.Module):
	'''A fully-connected ensemble classifier (no k-folds cross-validation)'''
	def __init__(self, input_size: int, hidden_size: int, output_size: int):
		super(EnsembleModel, self).__init__()
		self.roberta = RobertaModel.from_pretrained('roberta-base')
		roberta_hidden_size = self.roberta.config.hidden_size
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)
		self.logistic = nn.Linear(output_size + roberta_hidden_size, 1)
		
	def forward(self, data: dict, sentences: List[str], featurizer: Callable, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		outputs_roberta = self.roberta(**inputs).pooler_output
		features_tensor = torch.tensor(featurizer(sentences), dtype=torch.float).to(device)
		outputs_mlp = self.mlp(features_tensor)
		classifier_in = torch.cat((outputs_roberta, outputs_mlp), axis=1)
		logits = self.logistic(classifier_in)
		return logits

################################################################################################
###################	TRAIN BASE AND ENSEMBLE MODELS
################################################################################################

def train_model(
		model: Union[Regression, EnsembleModel, RoBERTaRegression, FeatureRegression], 
		sentences: List[str], labels: List[str], 
		test_sents: List[str], test_labels: List[str], 
		epochs: int, batch_size: int, lr: int,
		tokenizer: RobertaTokenizer, optimizer: torch.optim, loss_fn: Callable, measures: List[str], 
		device: str, regression: str, save_path: str = 'None', featurizer: Union[Callable] = lambda x: x):
	'''Train a neural network on linear regression
	NOTE: The input model must output just a single value
	'''
	
	# prepare model
	model.to(device)
	model.train()
	optim = optimizer(model.parameters(), lr=lr, weight_decay=1e-5)

	# load metrics
	metrics = []
	for metric in measures:
		m = load_metric(metric)
		metrics.append(m)

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)

	# create a dataset and dataloader to go iterate in batches
	dataset = FineTuneDataSet(shuffled_sentences, shuffled_labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	for epoch in range(epochs):
		for i, (batch, X, labels) in enumerate(dataloader):

			# send to the correct device
			y = torch.Tensor(labels)
			y = torch.unsqueeze(y, 1).to(device)

			# zero gradients
			optim.zero_grad()

			# get the output and do gradient descent
			output = model(batch, X, featurizer, device)

			loss = loss_fn(output, y)
			loss.backward()
			optim.step()

			# add batched results to metrics
			if regression == 'linear':
				pred_argmax = output
			else:
				pred_argmax = (output>0).float()
			y_argmax = torch.squeeze(y)
			for m in metrics:
				m.add_batch(predictions=torch.round(pred_argmax), references=y_argmax)
		
			if (i + 1) % 6 == 0:
		
				# output loss to standard output
				print(f'({epoch}, {(i + 1) * batch_size}) Loss: {loss.item()}', file = sys.stderr)

		# output metrics to standard output
		values = f"Training metrics:\n" # empty string 
		for m in metrics:
			val = m.compute()
			name = m.name if m.name != 'mse' else 'rmse'
			values += f"\t{name}: {val}\n"
		print(values, file = sys.stderr)

		evaluate_model(model, test_sents, test_labels, batch_size, featurizer, tokenizer, device, regression, measures, outfile=sys.stderr)

	# SAVE MODELS
	if save_path != 'None':
		try:
			print(f"Saving model to {save_path}/ensemble/...")
			torch.save(model, save_path + '/ensemble.pt')
		except:
			print(f"(Saving error) Couldn't save model to {save_path}/ensemble/...")


def evaluate_model(
	model: Union[Regression, EnsembleModel, RoBERTaRegression, FeatureRegression], 
	sentences: List[str], labels: List[str], batch_size: int, featurizer: Callable,
	tokenizer: RobertaTokenizer, device: str, regression: str, measures: List[str], outfile: Union[str, object]):
	'''Evaluate a regression model or Ensemble'''
	
	# get model ready
	model.to(device)
	model.eval()

	# load metrics
	metrics = []
	for metric in measures:
		m = load_metric(metric)
		metrics.append(m)

	# load the dataset
	dataset = FineTuneDataSet(sentences, labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	# intialize a list to store predictions
	predictions = []

	for batch, X, labels in dataloader:

		# send to the correct device
		y = torch.Tensor(labels)
		y = torch.unsqueeze(y, 1).to(device)

		output = model(batch, X, featurizer, device)

		# add batched results to metrics
		if regression == 'linear':
			pred_argmax = output
		else:
			pred_argmax = (output>0).float()
		y_argmax = torch.squeeze(y)
		for m in metrics:
			m.add_batch(predictions=torch.round(pred_argmax), references=y_argmax)

		# append predictions to list
		predictions.extend(torch.round(pred_argmax).clone().detach().to('cpu').tolist())
		
	# output metrics to standard output
	values = f"Evaluation metrics:\n" # empty string 
	for m in metrics:
		val = m.compute()
		name = m.name if m.name != 'mse' else 'rmse'
		values += f"\t{name}: {val}\n"
	print(values, file = sys.stderr)

	return predictions

################################################################################################
###################	TRAIN ENSEMBLE WITH KFOLDS MODEL
################################################################################################

def train_ensemble_linear(
		Transformer: RoBERTaRegression, FClassifier: FeatureRegression, Regressor: Regression, 
		tokenizer: RobertaTokenizer,
		sentences: List[str], labels: np.ndarray, featurizer: Callable,
		dev_sentences: List[str], dev_labels: np.ndarray,
		epochs: int, batch_size: int, 
		lr_transformer: float, lr_classifier: float, lr_regressor: float,
		kfolds: int,
		optimizer_transformer: torch.optim, optimizer_classifier: torch.optim, optimizer_regression: torch.optim,
		device: str, save_path: str = 'src/models/'
	):
	'''Train the ensemble on the training data'''

	# send the models to the device
	Transformer.to(device)
	FClassifier.to(device)
	Regressor.to(device)

	# initialize the optimizers for the Transformer and FClassifier
	optim_tr = optimizer_transformer(Transformer.parameters(), lr=lr_transformer, weight_decay=1e-5)
	optim_cl = optimizer_classifier(FClassifier.parameters(), lr=lr_classifier, weight_decay=1e-5)
	optim_log = optimizer_regression(Regressor.parameters(), lr=lr_regressor, weight_decay=1e-5)

	# set the loss function to MSE
	loss_fn = nn.MSELoss()

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)

	# prepare the kfolds cross validator
	# k-folds cross-validation trains the Transformer and FClassifier on parts of the
	# test data and then Regressor on the remaining dataset (outputs provided by the other two models)
	kfolder = StratifiedKFold(n_splits=kfolds)
	kfolder.get_n_splits(shuffled_sentences, shuffled_labels)

	for epoch in range(epochs):

		for fold, (train_i, test_i) in enumerate(kfolder.split(shuffled_sentences, shuffled_labels)):

			X = np.array(shuffled_sentences)
			y = np.array(shuffled_labels)

			X_models, y_models = X[train_i].tolist(), y[train_i].tolist()
			X_meta, y_meta = X[test_i].tolist(), y[test_i].tolist()

			# make the data a Dataset and put it in a DataLoader to batch it
			dataset = FineTuneDataSet(X_models, y_models)
			dataset.tokenize_data(tokenizer)
			dataloader = DataLoader(dataset, batch_size=batch_size)

			# TRAIN THE TRANSFORMER AND THE FCLASSIFIER
			Transformer.train()
			FClassifier.train()
			Regressor.train()

			for i, (batch, X, labels) in enumerate(dataloader):

				# TRANSFORMER TRAINING

				# send to the correct device
				labels = torch.unsqueeze(labels, 1)
				y = labels.to(device)

				# zero gradients
				optim.zero_grad()

				# get the output and do gradient descent
				output = Transformer(batch, X, featurizer, device)

				loss = loss_fn(output, y)
				loss.backward()
				optim.step()

				# FCLASSIFIER TRAINING

				# set classifier optimizer to 0-grad
				optim_cl.zero_grad()

				# initialize the feature tensor
				features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)

				# get feature classifier outputs
				classifier_outputs = FClassifier(features_tensor)

				loss_cl = loss_fn(classifier_outputs, y)
				loss_cl.backward()
				optim_cl.step()

			# make the data a Dataset and put it in a DataLoader to batch it
			dataset = FineTuneDataSet(X_meta, y_meta)
			dataset.tokenize_data(tokenizer)
			dataloader = DataLoader(dataset, batch_size=batch_size)

			# TRAIN THE LOGISTIC REGRESSOR
			Transformer.eval()
			FClassifier.eval()

			for i, (batch, X, labels) in enumerate(dataloader):

				optim_log.zero_grad()

				# GET TOTAL REGRESSION OUTPUT

				with torch.no_grad():
					# transformer value
					transformer_value = Transformer(batch, X, featurizer, device)
					
					# classifier value
					features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)
					feature_value = FClassifier(features_tensor)

				# TRAIN REGRESSOR
				all_values = torch.cat((transformer_value, feature_value), axis=1)
				output = Regressor(all_values)

				# backpropagate through regressor
				loss_lg = loss_logistic(output, y)
				loss_lg.backward()
				optim_log.step()
				
			# Get the accuracy of the fold on the test data
			print(f"Fold {fold} accuracies:", file = sys.stderr)
			evaluate_ensemble_linear(Transformer, FClassifier, Regressor, tokenizer, dev_sentences, dev_labels, featurizer, batch_size, device, sys.stderr)

		# Get the accuracy of each epoch on the test data
		print(f"Epoch {epoch} accuracies:", file = sys.stderr)
		evaluate_ensemble_linear(Transformer, FClassifier, Regressor, tokenizer, dev_sentences, dev_labels, featurizer, batch_size, device, sys.stderr)


def evaluate_ensemble_linear(
		Transformer: RobertaModel, FClassifier: FeatureClassifier, Regressor: Regression, tokenizer: RobertaTokenizer,
		sentences: List[str], labels: np.ndarray, featurizer: Callable, batch_size: int, device: str, file: object = sys.stdout
	):
	'''Evaluate the Ensemble'''
	
	# send the models to the device
	Transformer.to(device)
	FClassifier.to(device)

	# turn the models into eval mode
	Transformer.eval()
	FClassifier.eval()
	Regressor.eval()

	rmse = load_metric(metric)
	tr_rmse = load_metric(metric)
	cl_rmse = load_metric(metric)

	# make the data a Dataset and put it in a DataLoader to batch it
	dataset = FineTuneDataSet(sentences, labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	# intialize a list to store predictions
	predictions, roberta_preds, feature_preds = [], [], []

	for batch, X in dataloader:

		# send labels to device
		y = torch.reshape(batch['labels'], (batch['labels'].size()[0], 1)).float().to(device)

		with torch.no_grad():
			# transformer value
			transformer_value = Transformer(batch, X, featurizer, device)
			
			# feature value
			features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)
			feature_value = FClassifier(features_tensor)

		with torch.no_grad():
			# transformer output
			transformer_out = Transformer(batch, device)
			
			# features ouput
			features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)
			featurer_out = FClassifier(features_tensor)
			
			# OUTPUT EACH Regressor'S RESULT INDIVIDUALLY
			for m1 in tr_metrics:
				m1.add_batch(predictions=transformer_out, references=y)
			for m2 in cl_metrics:
				m2.add_batch(predictions=featurer_out, references=y)

			# EVALUATE REGRESSOR
			all_out = torch.cat((transformer_out, featurer_out), axis=1)
			y_hats = Regressor(all_out)

			# add batch to output
			as_list = y_hats.to('cpu').tolist()
			roberta_preds.extend(transformer_out.to('cpu').tolist())
			feature_preds.extend(featurer_out.to('cpu').tolist())
			predictions.extend(as_list)

		# add batched results to metrics
		rmse.add_batch(predictions=y_hats, references=y)

	# output metrics to standard output
	val_en, val_tr, val_cl = f"Evaluation:", f"Evaluation:", f"Evaluation:" 
	for m1, m2, m3 in zip(rmse, tr_rmse, cl_rmse):
		val1, val2, val3 = m1.compute(), m2.compute(), m3.compute()
		val_en += f"\t Ensemble RMSE: {val1}\n"
		val_tr += f"\t Transformer RMSE: {val2}\n"
		val_cl += f"\t Featurizer RMSE: {val3}\n"
	print("\n".join([val_en, val_tr, val_cl]), file = file)
	return predictions, roberta_preds, feature_preds