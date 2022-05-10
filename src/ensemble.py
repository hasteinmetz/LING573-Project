import json
import utils
import torch
import argparse
import sys
import numpy as np
import pandas as pd
from typing import *
from featurizer import featurize, TFIDFGenerator
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import RobertaForSequenceClassification, BatchEncoding, RobertaConfig, RobertaTokenizer, get_scheduler
from datasets import load_metric
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from math import ceil

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


class Ensemble(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, output_size: int):
		super(Ensemble, self).__init__()
		self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size+1)
		)
		self.output = nn.Linear(2+2, output_size)

	def forward(self, data: dict, sentences: List[str], featurizer: Callable, device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		outputs_roberta = self.roberta(**inputs).logits
		features_tensor = torch.tensor(featurizer(sentences), dtype=torch.float).to(device)
		outputs_mlp = self.mlp(features_tensor)
		classifier_in = torch.cat((outputs_roberta, outputs_mlp), axis=1)
		logits = self.output(classifier_in)
		return logits

def train(model: Ensemble, sentences: List[str], labels: List[str], epochs: int, batch_size: int, lr: int,
	featurizer: Callable, tokenizer: RobertaTokenizer, optimizer: torch.optim, loss_fn: Callable, device: str):
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

	dataset = FineTuneDataSet(shuffled_sentences, shuffled_labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	for epoch in range(epochs):
		for i, (batch, X) in enumerate(dataloader):

			y = torch.reshape(batch['labels'], (batch['labels'].size()[0], 1)).float().to(device)

			optim.zero_grad()

			output = model(batch, X, featurizer, device)

			loss = loss_fn(output, y)
			loss.backward()
			optim.step()

			# add batched results to metrics
			pred_argmax = torch.round(torch.sigmoid(output))
			for m in metrics:
				m.add_batch(predictions=pred_argmax, references=y)
        
			if (i + 1) % 6 == 0:
		
				# output metrics to standard output
				print(f'({epoch}, {(i + 1) * batch_size}) Loss: {loss.item()}', file = sys.stderr)

		# output metrics to standard output
		values = f"" # empty string 
		for m in metrics:
			val = m.compute()
			values += f"{m.name}:\n\t {val}\n"
		print(values, file = sys.stderr)


def evaluate(model: Ensemble, sentences: List[str], labels: List[str], batch_size: int, lr: int, 
	tokenizer: RobertaTokenizer, featurizer: Callable, device: str):
	'''Train the Ensemble neural network'''
	model.to(device)
	model.eval()

	metrics = []
	for metric in ['f1', 'accuracy']:
		m = load_metric(metric)
		metrics.append(m)

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)

	dataset = FineTuneDataSet(shuffled_sentences, shuffled_labels)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	predictions = []

	for batch, X in dataloader:

		y = torch.reshape(batch['labels'], (batch['labels'].size()[0], 1)).float().to(device)

		output = model(batch, X, featurizer, device)

		# add batch to output
		pred_argmax = torch.round(torch.sigmoid(output))
		as_list = pred_argmax.clone().detach().to('cpu').tolist()
		predictions.append(as_list)

		# add batched results to metrics
		for m in metrics:
			m.add_batch(predictions=pred_argmax, references=y)
	
	# output metrics to standard output
	print(f'Loss: {loss.item()}', file = sys.stderr)

	# output metrics to standard output
	values = f"" # empty string 
	for m in metrics:
		val = m.compute()
		values += f"{m.name}:\n\t {val}\n"
	print(values, file = sys.stderr)
	return predictions


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
	train_sentences, train_labels = utils.read_data_from_file(args.train_data_path)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_data_path)

	# initialize tf-idf vectorizer
	tfidf = TFIDFGenerator(train_sentences, 'english', train_labels)
	featurizer = lambda x: featurize(x, tfidf)

	# get input size
	input_size = featurizer(train_sentences[0:2]).shape[1]

	# initialize ensemble model
	print("initializing ensemble architecture")
	OPTIMIZER = AdamW
	LR = 5e-5
	BATCH_SIZE = 32
	LOSS = nn.BCEWithLogitsLoss()
	EPOCHS = 1
	TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
	model = Ensemble(input_size, 100, 1)

	# train the model
	train(model, train_sentences, train_labels, EPOCHS, BATCH_SIZE, LR, featurizer, TOKENIZER, OPTIMIZER, LOSS, device)
	preds = evaluate(model, dev_sentences, dev_labels, BATCH_SIZE, TOKENIZER, featurizer, device)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', help="path to input training data file")
	parser.add_argument('--dev_data_path', help="path to input dev data file")
	parser.add_argument('--output_file', help="path to output data file")
	args = parser.parse_args()

	main(args)
