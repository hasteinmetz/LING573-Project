import torch
from typing import *
from transformers import RobertaTokenizer

class FineTuneDataSet(Dataset):
    '''Class creates a list of dicts of sentences and labels
    and behaves like a list but also stores sentences and labels for
    future use'''
    def __init__(self, sentences: List[str], labels: List[int]):
        self.sentences = sentences
        self.labels = labels

    def tokenize_data(self, tokenizer: RobertaTokenizer):
        if not hasattr(self, 'encodings'):
            # encode the data
            self.encodings = tokenizer(self.sentences, return_tensors="pt", padding=True)
            self.input_ids = self.encodings['input_ids']

    def __getitem__(self, index: int,):
        if not hasattr(self, 'encodings'):
            raise AttributeError("Did not initialize encodings or input_ids")
        else:
            item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
            item['label'] = torch.tensor(self.labels[index])
            return item

    def __len__(self):
        return len(self.labels)
