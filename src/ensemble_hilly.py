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
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_metric
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import json

nn = torch.nn

class FineTuneDataSet(Dataset):
	'''Class creates a list of dicts of sentences and labels
	and behaves list a list but also stores sentences and labels for
	future use'''
	def __init__(self, sentences: List[str], labels: List[int]):
		self.sentences = sentences
		self.labels = labels

	def tokenize_data(self, tokenizer: RobertaTokenizer):
		if not hasattr(self, 'encodings'):
			# encode the data
			self.encodings = tokenizer(self.sentences, return_tensors="pt", padding=True)
			self.input_ids = self.encodings['input_ids']

	def __getitem__(self, index: int):
		if not hasattr(self, 'encodings'):
			raise AttributeError("Did not initialize encodings or input_ids")
		else:
			item = {key: val[index].clone().detach() for key, val in self.encodings.items()}
			item['labels'] = torch.tensor(self.labels[index])
			return item, self.sentences[index]

	def __len__(self):
		return len(self.labels)


def expand_labels(arr: np.ndarray) -> torch.Tensor:
	'''Expand the array labels so that it's a [size, no_labels] matrix'''
	labels = set(arr.tolist())
	print('labels and len', labels, len(labels))
	new_arr = np.zeros((arr.shape[0], len(labels)), dtype=float)
	for i in range(len(arr)):
		new_arr[i, int(arr[i])] = 1.0
	return new_arr


class Ensemble(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, output_size: int):
		super(Ensemble, self).__init__()
		self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)
		self.logistic = nn.Linear(output_size * 2, output_size)

	def forward(self, data: dict, sentences: List[str], featurizer: Callable, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		outputs_roberta = self.roberta(**inputs).logits
		features_tensor = torch.tensor(featurizer(sentences), dtype=torch.float).to(device)
		outputs_mlp = self.mlp(features_tensor)
		classifier_in = torch.cat((outputs_roberta, outputs_mlp), axis=1)
		logits = self.logistic(classifier_in)
		return logits


def train_ensemble(model: Ensemble, 
		sentences: List[str], labels: List[str], 
		test_sents: List[str], test_labels: List[str], 
		epochs: int, batch_size: int, lr: int,
		featurizer: Callable, tokenizer: RobertaTokenizer, 
		optimizer: torch.optim, loss_fn: Callable, device: str, save_path: str):
	'''Train the Ensemble neural network'''
	
	model.to(device)
	model.train()
	optim = optimizer(model.parameters(), lr=lr, weight_decay=1e-5)

	metrics = []
	for metric in ['f1', 'accuracy']:
		m = load_metric(metric)
		metrics.append(m)

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)
	labels_arr = expand_labels(shuffled_labels)

	# create a dataset and dataloader to go iterate in batches
	dataset = FineTuneDataSet(shuffled_sentences, labels_arr)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	for epoch in range(epochs):
		for i, (batch, X) in enumerate(dataloader):

			# send to the correct device
			y = batch['labels'].to(device)

			optim.zero_grad()

			output = model(batch, X, featurizer, device)

			loss = loss_fn(output, y)
			loss.backward()
			optim.step()

			# add batched results to metrics
			pred_argmax = torch.argmax(torch.sigmoid(output), dim=-1)
			for m in metrics:
				m.add_batch(predictions=pred_argmax, references=torch.argmax(y, dim=-1))
		
			if (i + 1) % 6 == 0:
		
				# output metrics to standard output
				print(f'({epoch}, {(i + 1) * batch_size}) Loss: {loss.item()}', file = sys.stderr)

		# output metrics to standard output
		values = f"" # empty string 
		for m in metrics:
			val = m.compute()
			values += f"{m.name}:\n\t {val}\n"
		print(values, file = sys.stderr)

		evaluate(model, test_sents, test_labels, batch_size, tokenizer, featurizer, device, outfile=sys.stderr)

	# SAVE MODELS
	try:
		print("Saving model to {save_path}/ensemble/...")
		torch.save(Ensemble, save_path + '/ensemble/')
	except:
		print(f"(Saving error) Couldn't save model to {save_path}/ensemble/...")

def evaluate(model: Ensemble, sentences: List[str], labels: List[str], batch_size: int,
	tokenizer: RobertaTokenizer, featurizer: Callable, device: str, outfile: Union[str, object]):
	'''Train the Ensemble neural network'''
	model.to(device)
	model.eval()

	metrics = []
	for metric in ['f1', 'accuracy']:
		metrics.append(load_metric(metric))
		
	# convert labels to the correct shape
	labels_arr = expand_labels(labels)

	dataset = FineTuneDataSet(sentences, labels_arr)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	# intialize a list to store predictions
	predictions = []

	for batch, X in dataloader:

		# send to the correct device
		y = batch['labels'].to(device)

		logits = model(batch, X, featurizer, device)

		# add batch to output
		predicted = torch.argmax(torch.sigmoid(logits), dim=-1)
		
		# append predictions to list
		predictions.extend(predicted.clone().detach().to('cpu').tolist())

		# add batched results to metrics
		for m in metrics:
			m.add_batch(predictions=predicted, references=torch.argmax(y, dim=-1))
		
	# output metrics to standard output
	val = f"" # empty string
	for m in zip(metrics):
		val = m.compute()
		val += f"\t{m.name}: {val}\n"
	print(val, file = outfile)

	return predictions


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

	# load PCA and extract principle components from the training data
	pca = PCA()
	training_feature_matrix = featurize(train_sentences, train_labels, hurtlex_dict, hurtlex_feat_list, tfidf)
	pca.fit(training_feature_matrix)
	print(f"Fitted PCA. Previously there were {training_feature_matrix.shape[1]} " + 
		f"features. Now there are {pca.n_components_} features.", file=sys.stderr)

	# reduce the parameters of the featurize function
	FEATURIZER = lambda x: featurize(x, train_labels, hurtlex_dict, hurtlex_feat_list, tfidf, pca)

	# get input size
	input_size = FEATURIZER(train_sentences[0:1]).shape[1]
	
	# LOAD CONFIGURATION
	config_file = f'src/configs/ensemble-{args.job}.json'
	with open(config_file, 'r') as f1:
		configs = f1.read()
		train_config = json.loads(configs)

	# load important parts of the model
	print("Initializing ensemble architecture...\n")
	TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")

	# load model is already available
	if args.reevaluate != 1:
		print("Training model...\n")

		# initialize ensemble model
		ENSEMBLE = Ensemble(input_size, 100, 2)
		OPTIMIZER = AdamW
		LOSS = nn.CrossEntropyLoss()
	
		# train the model
		train_ensemble(
			model=ENSEMBLE, 
			sentences=train_sentences, 
			labels=train_labels,
			test_sents=dev_sentences, 
			test_labels=dev_labels,
			featurizer=FEATURIZER, 
			tokenizer=TOKENIZER, 
			optimizer=OPTIMIZER,
			loss_fn=LOSS, 
			device=DEVICE, 
			save_path = f"{args.model_save_path}/{args.job}",
			**train_config
		)
	else:
		try:
			print("Loading existing model...\n")
			# TODO: Change the directory organization
			model = torch.load(f'src/models/testing-ensemble/{args.job}/')
		except ValueError:
			print("No existing model found. Rerun without --reevaluate")


	# evaluate the model on test data
	print("Evaluating models...")
	preds = evaluate(
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
	output_file = f'{args.output_path}/{args.job}/ensemble-output.csv'
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
