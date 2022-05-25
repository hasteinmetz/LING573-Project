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

nn = torch.nn
class RoBERTa(nn.Module):
	'''A wrapper around the RoBERTa model with a defined forward function'''
	def __init__(self):
		super(RoBERTa, self).__init__()
		self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')

	def forward(self, data: dict, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		roberta_out = self.roberta(**inputs)
		return roberta_out
class FeatureClassifier(nn.Module):
	'''Simple feed forward neural network to classify a word's lexical features'''
	def __init__(self, input_size: int, hidden_size: int, num_classes: int):
		super(FeatureClassifier, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(0.75), # high-ish dropout to avoid overfitting to certain features
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(0.75),
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
		return torch.sigmoid(linear)

def train_ensemble(
		Transformer: RoBERTa, FClassifier: FeatureClassifier, LogRegressor: LogisticRegression, tokenizer: RobertaTokenizer,
		sentences: List[str], labels: np.ndarray, featurizer: Callable,
		dev_sentences: List[str], dev_labels: np.ndarray,
		epochs: int, batch_size: int, 
		lr_transformer: float, lr_classifier: float, lr_regressor: float,
		kfolds: int,
		optimizer_transformer: torch.optim, optimizer_classifier: torch.optim, optimizer_regression: torch.optim,
		loss_fn: Callable, device: str, save_path: str = 'src/models/'
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
	loss_logistic = nn.BCELoss()

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

			# turn into [n, 2] matrices
			y_models = make_labels_binary(y_models)
			y_meta = make_labels_binary(y_meta)

			# make the data a Dataset and put it in a DataLoader to batch it
			dataset = FineTuneDataSet(X_models, y_models, verbose = 'verbose')
			dataset.tokenize_data(tokenizer)
			dataloader = DataLoader(dataset, batch_size=batch_size)

			# TRAIN THE TRANSFORMER AND THE FCLASSIFIER
			Transformer.train()
			FClassifier.train()
			LogRegressor.train()

			for i, (batch, X) in enumerate(dataloader):

				# change the shape of labels to [n, 2]
				y = batch['labels'].to(device)

				# TRANSFORMER TRAINING

				# set transformer optimizer to 0-grad
				optim_tr.zero_grad()

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
					# transformer logits
					transformer_logits = Transformer(batch, device).logits
					
					# classifier logits
					features_tensor = torch.tensor(featurizer(X), dtype=torch.float).to(device)
					feature_logits = FClassifier(features_tensor)

				# TRAIN REGRESSOR
				all_logits = torch.cat((transformer_logits, feature_logits), axis=1)
				output = LogRegressor(all_logits)
				
				# reshape true labels
				true_ys = torch.argmax(batch['labels'], dim = 1)
				y = torch.reshape(true_ys, (true_ys.size()[0], 1)).float().to(device)

				# get loss and do backpropagation
				loss_lg = loss_logistic(output, y)
				loss_lg.backward()
				optim_log.step()

				if (i + 1) % 20 == 0:
			
					# output metrics to standard output
					correct = (torch.round(output) == y).type(torch.float).sum().item() 
					total = output.shape[0]
					accuracy = correct/total
					print(
						f'(epoch {epoch+1}, fold {fold+1}, samples {(i+1)*batch_size}) ' +
						f'Regression Accuracy: {accuracy}, Loss: {loss_lg.item()}',
						file = sys.stderr
					)
				
			# Get the accuracy of the fold on the test data
			print(f"Fold {fold} accuracies:", file = sys.stderr)
			evaluate_ensemble(Transformer, FClassifier, LogRegressor, tokenizer, dev_sentences, dev_labels, featurizer, batch_size, device, sys.stderr)

		# Get the accuracy of each epoch on the test data
		print(f"Epoch {epoch} accuracies:", file = sys.stderr)
		evaluate_ensemble(Transformer, FClassifier, LogRegressor, tokenizer, dev_sentences, dev_labels, featurizer, batch_size, device, sys.stderr)

	# SAVE MODELS
	try:
		print("Saving models to /src/models/kfold-ensemble/...")
		torch.save(Transformer, save_path + 'roberta')
		torch.save(FClassifier, save_path + 'featurizer')
		torch.save(LogRegressor, save_path + 'regression')
	except:
		print("(Saving error) Saving models to src/models/kfold-ensemble/...")
		torch.save(Transformer, 'src/models/tmp_roberta')
		torch.save(FClassifier, 'src/models/tmp_featurizer')
		torch.save(LogRegressor, 'src/models/tmp_regression')

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

	# convert the model labels to a [n, 2] matrix (needed for roberta)
	labels_arr = make_labels_binary(labels)

	# make the data a Dataset and put it in a DataLoader to batch it
	dataset = FineTuneDataSet(sentences, labels_arr, verbose = 'verbose')
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	# intialize a list to store predictions
	predictions, roberta_preds, feature_preds = [], [], []

	for batch, X in dataloader:

		# send labels to device
		y = batch['labels'].to(device)

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
		y_argmax = torch.argmax(y, dim = -1)
		for m1 in tr_metrics:
			m1.add_batch(predictions=tr_argmax, references=y_argmax)
		for m2 in cl_metrics:
			m2.add_batch(predictions=cl_argmax, references=y_argmax)

		# EVALUATE REGRESSOR
		all_logits = torch.cat((transformer_logits, feature_logits), axis=1)
		y_hats = LogRegressor(all_logits)

		# add batch to output
		as_list = torch.round(y_hats).to('cpu').tolist()
		roberta_preds.extend(tr_argmax.to('cpu').tolist())
		feature_preds.extend(cl_argmax.to('cpu').tolist())
		predictions.extend(as_list)

		# add batched results to metrics
		for m3 in metrics:
			m3.add_batch(predictions=torch.round(y_hats), references=y_argmax)

	# output metrics to standard output
	val_en, val_tr, val_cl = f"", f"", f"" # empty strings
	for m1, m2, m3 in zip(metrics, tr_metrics, cl_metrics):
		val1, val2, val3 = m1.compute(), m2.compute(), m3.compute()
		val_en += f"Ensemble\t{m1.name}: {val1}\n"
		val_tr += f"Transformer\t{m2.name}: {val2}\n"
		val_cl += f"Featurizer\t{m3.name}: {val3}\n"
	print("\n".join([val_en, val_tr, val_cl]), file = file)
	return predictions, roberta_preds, feature_preds


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

	print("reducing feature dimensions...")
	if args.dim_reduc_method == 'pca':
		train_feature_vector = featurize(train_sentences, hurtlex_dict, hurtlex_feat_list, tfidf)
		train_pca = PCA(.95)
		train_pca.fit(train_feature_vector)
		print("\tnum components: {}".format(train_pca.n_components))
		FEATURIZER = lambda x: train_pca.transform(featurize(x, hurtlex_dict, hurtlex_feat_list, tfidf))
	else:
		train_feat_vector = featurize(train_sentences, hurtlex_dict, hurtlex_feat_list)
		train_feature_vector, feat_indices = k_perc_best_f(train_feat_vector, train_labels, 70)
		# use the features inside the model using a featurize function
		FEATURIZER = lambda x: prune_test(featurize(x, hurtlex_dict, hurtlex_feat_list, tfidf), feat_indices)

	# get input size
	input_size = FEATURIZER(train_sentences[0:1]).shape[1]

	# initialize ensemble model
	# TODO: MAKE ADD THIS TO A CONFIG FILE INSTEAD
	print("Initializing ensemble architecture...\n")
	OPTIMIZER_TRANSFORMER = AdamW
	OPTIMIZER_CLASSIFIER = Adagrad
	OPTIMIZER_REGRESSOR = SGD
	LR_TRANSFORMER = 5e-5
	LR_CLASSIFIER = 8e-3
	LR_REGRESSOR = 1e-2
	BATCH_SIZE = 32
	LOSS = nn.CrossEntropyLoss()
	EPOCHS = 1
	# TODO: CONFIGURE DROPOUT RATES FOR ROBERTA TO AVOID OVERFITTING
	# TODO: LOAD MODEL IF AVAILABLE
	TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
	ROBERTA = RoBERTa()
	FEATURECLASSIFIER = FeatureClassifier(input_size, 100, 2)
	LOGREGRESSION = LogisticRegression(4)
	KFOLDS = 5

	# train the model
	train_ensemble(
		ROBERTA, FEATURECLASSIFIER, LOGREGRESSION, TOKENIZER,
		train_sentences, train_labels, FEATURIZER, 
		dev_sentences, dev_labels,
		EPOCHS, BATCH_SIZE, 
		LR_TRANSFORMER, LR_CLASSIFIER, LR_REGRESSOR,
		KFOLDS,
		OPTIMIZER_TRANSFORMER, OPTIMIZER_CLASSIFIER, OPTIMIZER_REGRESSOR,
		LOSS, DEVICE, save_path = args.model_save_path
	)

	# evaluate the model on test data
	print("Evaluating models...")
	preds, robs, feats = evaluate_ensemble(
		ROBERTA, FEATURECLASSIFIER, LOGREGRESSION, TOKENIZER,
		dev_sentences, dev_labels, FEATURIZER, BATCH_SIZE, DEVICE
	)

	# write results to output file
	dev_out_d = {'sentence': dev_sentences, 'predicted': preds, 'transformer': robs, 'featurizer': feats, 'correct_label': dev_labels}
	dev_out = pd.DataFrame(dev_out_d)
	output_file = f'{args.output_path}/{args.job}/fusion-output.csv'
	dev_out.to_csv(output_file, index=False, encoding='utf-8')

	# filter the data so that only negative examples are there
	data_filtered = dev_out.loc[~(dev_out['predicted'] == dev_out['correct_label'])]
	error_file = f'{args.error_path}-{args.job}.csv'
	data_filtered.to_csv(error_file, index=False, encoding='utf-8')

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', help="path to input training data file")
	parser.add_argument('--dev_data_path', help="path to input dev data file")
	parser.add_argument('--hurtlex_path', help="path to hurtlex dictionary")
	parser.add_argument('--output_path', help="path to output data file")
	parser.add_argument('--dim_reduc_method', help="method used to reduce the dimensionality of feature vectors", default = 'pca')
	parser.add_argument('--model_save_path', help="path to save models")
	parser.add_argument('--error_path', help="path to save error analysis")
	parser.add_argument('--job', help="job being done (humor or controversy) to help with folder organization")
	parser.add_argument('--debug', help="debug the ensemble with small dataset", type=int)
	parser.add_argument('--index', help="column to select from data", type=int)
	args = parser.parse_args()

	main(args)
