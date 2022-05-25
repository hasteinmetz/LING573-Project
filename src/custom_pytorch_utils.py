import torch
from typing import *
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import numpy as np

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
			# need this line for sequence classification item['labels'] = torch.tensor(self.labels[index])
			return item, self.sentences[index], torch.tensor(self.labels[index])

	def __len__(self):
		return len(self.labels)


def make_torch_labels_binary(arr: np.ndarray) -> np.ndarray:
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