import torch
from typing import *
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import numpy as np

class FineTuneDataSet(Dataset):
	'''Class creates a list of dicts of sentences and labels
	and behaves list a list but also stores sentences and labels for
	future use
		Parameters:
			- sentences: the training/test sentences
			- labels: the training/test labels
			- verbose: whether to output the sentences or not
			- encodings/input_ids: Huggingface dataset and pytorch compatibility
	'''
	def __init__(self, sentences: List[str], labels: List[int], verbose: str = 'no'):
		self.sentences = sentences
		self.labels = labels
		self.verbose = verbose

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
			if self.verbose == 'verbose':
				return item, self.sentences[index]
			else:
				return item

	def __len__(self):
		return len(self.labels)

def make_torch_labels_binary(labels: np.ndarray) -> torch.tensor:
	'''Helper function that turns [n x 1] labels into [n x 2] labels'''
	zeros = torch.zeros((labels.shape[0], 2), dtype=float)
	for i in range(len(labels)):
		zeros[i, int(labels[i])] = 1.0
	return zeros

def make_labels_binary(arr: np.ndarray) -> np.ndarray:
	'''Expand the array labels in a one-dim array so that it's a [size, no_labels] matrix.
	You can use the regular cross entropy function on this sort of data 
	i.e. something like:
			[
				[0, 1],
				[1, 0], ...
			]
	'''
	labels = set(arr.tolist())
	new_arr = np.zeros((arr.shape[0], len(labels)), dtype=float)
	for i in range(len(arr)):
		new_arr[i, int(arr[i])] = 1.0
	return new_arr