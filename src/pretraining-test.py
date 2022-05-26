#!/usr/bin/env python

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
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification as RobertaSeqCls
from datasets import load_metric
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import json

nn = torch.nn

class PretrainFineTuneDataSet(Dataset):
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
			# item['labels'] = torch.tensor(self.labels[index])
			return item, self.sentences[index], torch.tensor(self.labels[index])

	def __len__(self):
		return len(self.labels)


def expand_labels(arr: np.ndarray) -> torch.Tensor:
	'''Expand the array labels so that it's a [size, no_labels] matrix'''
	labels = set(arr.tolist())
	new_arr = np.zeros((arr.shape[0], len(labels)), dtype=float)
	for i in range(len(arr)):
		new_arr[i, int(arr[i])] = 1.0
	return new_arr
	

class Regression(nn.Module):
	def __init__(self):
		super(Regression, self).__init__()
		self.roberta = RobertaModel.from_pretrained('roberta-base')
		roberta_hidden_size = self.roberta.config.hidden_size
		self.regression = nn.Linear(roberta_hidden_size, 1)

	def forward(self, data: dict, sentences: List[str], device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		outputs_roberta = self.roberta(**inputs).pooler_output
		pred = self.regression(outputs_roberta)
		return pred


class Classifier(nn.Module):
	def __init__(self, roberta):
		super(Classifier, self).__init__()
		init_roberta = RobertaSeqCls.from_pretrained('roberta-base')
		roberta.pooler = None
		init_roberta.roberta = roberta
		self.roberta = init_roberta

	def forward(self, data: dict, sentences: List[str], device: str):
		# tokenize the data
		inputs = {k:v.to(device) for k,v in data.items()}
		outputs_roberta = self.roberta(**inputs).logits
		return outputs_roberta


def train_model(model: Union[Regression, RobertaSeqCls], 
		sentences: List[str], labels: List[str], 
		test_sents: List[str], test_labels: List[str], 
		epochs: int, batch_size: int, lr: int,
		tokenizer: RobertaTokenizer, 
		optimizer: torch.optim, loss_fn: Callable, device: str, cl: str, save_path: str):
	'''Train the neural network'''
	
	model.to(device)
	model.train()
	optim = optimizer(model.parameters(), lr=lr, weight_decay=1e-5)

	metrics = []
	if cl == 'yes':
		measurements = ['f1', 'accuracy']
		for metric in measurements:
			m = load_metric(metric)
			metrics.append(m)
	else:
		m = load_metric('mse', squared=False)
		metrics = [m]

	# shuffle the data
	shuffled_sentences, shuffled_labels = shuffle(sentences, labels, random_state = 0)
	if cl == 'yes':
		labels_arr = expand_labels(shuffled_labels)
	else:
		labels_arr = shuffled_labels

	# create a dataset and dataloader to go iterate in batches
	dataset = PretrainFineTuneDataSet(shuffled_sentences, labels_arr)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	for epoch in range(epochs):
		for i, (batch, X, labels) in enumerate(dataloader):

			if cl == 'yes':
				batch['labels'] = labels
			else:
				labels = torch.unsqueeze(labels, 1)

			# send to the correct device
			y = labels.to(device)

			optim.zero_grad()

			output = model(batch, X, device)

			loss = loss_fn(output, y)
			loss.backward()
			optim.step()

			# add batched results to metrics
			# if and else clause used to distinguish classifier from regressor
			if cl == 'yes':
				pred_argmax = torch.argmax(torch.sigmoid(output), dim=-1)
				y_argmax = torch.argmax(y, dim=-1)
			else:
				pred_argmax = output
				y_argmax = torch.squeeze(y)
			for m in metrics:
				m.add_batch(predictions=pred_argmax, references=y_argmax)
		
			if (i + 1) % 6 == 0:
		
				# output metrics to standard output
				print(f'({epoch}, {(i + 1) * batch_size}) Loss: {loss.item()}', file = sys.stderr)

		# output metrics to standard output
		values = f"Training metrics:\n" # empty string 
		for m in metrics:
			if m.name == 'mse':
				val = m.compute(squared=False)
				name = 'rmse'
			else:
				val = m.compute()
				name = m.name
			values += f"\t{name}: {val}\n"
		print(values, file = sys.stderr)

		evaluate(model, test_sents, test_labels, batch_size, tokenizer, device, cl, outfile=sys.stderr)

	# SAVE MODELS
	if cl == 'yes':
		try:
			print(f"Saving model to {save_path}...")
			torch.save(model, save_path + '/model.pt')
		except:
			print(f"(Saving error) Couldn't save model to {save_path}/model.pt")


def evaluate(model, sentences: List[str], labels: List[str], batch_size: int,
	tokenizer: RobertaTokenizer, device: str, cl: str, outfile: Union[str, object]):
	'''Train the pretrained neural network'''
	model.to(device)
	model.eval()

	metrics = []
	if cl == 'yes':
		measurements = ['f1', 'accuracy']
		for metric in measurements:
			m = load_metric(metric)
			metrics.append(m)
	else:
		m = load_metric('mse', squared=False)
		metrics = [m]

	# shuffle the data
	if cl == 'yes':
		labels_arr = expand_labels(labels)
	else:
		labels_arr = labels

	dataset = PretrainFineTuneDataSet(sentences, labels_arr)
	dataset.tokenize_data(tokenizer)
	dataloader = DataLoader(dataset, batch_size=batch_size)

	# intialize a list to store predictions
	predictions = []

	for batch, X, labels in dataloader:

		if cl == 'yes':
			batch['labels'] = labels
		else:
			labels = torch.unsqueeze(labels, 1)

		# send to the correct device
		y = labels.to(device)

		output = model(batch, X, device)

		# add batched results to metrics
		# if and else clause used to distinguish classifier from regressor
		if cl == 'yes':
			pred_argmax = torch.argmax(torch.sigmoid(output), dim=-1)
			y_argmax = torch.argmax(y, dim=-1)
			predictions.extend(pred_argmax.clone().detach().to('cpu').tolist())
		else:
			pred_argmax = output
			y_argmax = torch.squeeze(y)
		for m in metrics:
			m.add_batch(predictions=pred_argmax, references=y_argmax)
		
	# output metrics to standard output
	values = f"Evaluation metrics:\n" # empty string
	for m in metrics:
		if m.name == 'mse':
			val = m.compute(squared=False)
			name = 'rmse'
		else:
			val = m.compute()
			name = m.name
		values += f"\t{name}: {val}\n"
	print(values, file = outfile)

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
	
	# LOAD CONFIGURATION
	config_file = f'src/configs/pretrain-{args.job}.json'
	with open(config_file, 'r') as f1:
		configs = f1.read()
		train_config = json.loads(configs)

	# load important parts of the model
	print("Initializing pretraining-test architecture...\n")
	TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")

	# load model is already available
	if args.reevaluate != 1:
		print("Training model...\n")

		# initialize the model thingies
		REGRESSOR = Regression()
		OPTIMIZER = AdamW
		LOSS_RE = nn.MSELoss()
		LOSS_CL = nn.CrossEntropyLoss()
	
		# pretrain the model on the regression task
		train_model(
			model=REGRESSOR, 
			sentences=pt_sentences,  
			labels=pt_labels,
			test_sents=dev_pt_sentences, 
			test_labels=dev_pt_labels,
			tokenizer=TOKENIZER, 
			optimizer=OPTIMIZER,
			loss_fn=LOSS_RE, 
			device=DEVICE, 
			cl='no',
			save_path = f"{args.model_save_path}/{args.job}",
			**train_config
		)

		# reload the model
		CLASSIFIER = Classifier(REGRESSOR.roberta)

		# finetune the model on the regression task
		train_model(
			model=CLASSIFIER, 
			sentences=train_sentences, 
			labels=train_labels,
			test_sents=dev_sentences, 
			test_labels=dev_labels,
			tokenizer=TOKENIZER, 
			optimizer=OPTIMIZER,
			loss_fn=LOSS_CL, 
			device=DEVICE, 
			cl='yes',
			save_path = f"{args.model_save_path}/{args.job}",
			**train_config
		)
	else:
		try:
			print("Loading existing model...\n")
			# TODO: Change the directory organization
			model = torch.load(f'src/models/testing-pretraining/{args.job}/')
		except ValueError:
			print("No existing model found. Rerun without --reevaluate")


	# evaluate the model on test data
	print("Evaluating models...")
	preds = evaluate(
		model=CLASSIFIER, 
		sentences=dev_sentences, 
		labels=dev_labels, 
		batch_size=train_config['batch_size'],
		tokenizer=TOKENIZER, 
		device=DEVICE, 
		cl='yes',
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
	parser.add_argument('--job', help="to help name files when running batches", default='test', type=str)
	parser.add_argument('--output_path', help="path to output data file")
	parser.add_argument('--model_save_path', help="path to save models")
	parser.add_argument('--error_path', help="path to save error analysis")
	parser.add_argument('--debug', help="debug the pretraining with small dataset", type=int)
	parser.add_argument('--index', help="column to select from data", type=int)
	parser.add_argument('--reevaluate', help="use 1 to reload an existing model if already completed", type=int, default=0)
	args = parser.parse_args()

	main(args)
