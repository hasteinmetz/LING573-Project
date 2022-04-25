#!/usr/bin/env python

'''
PyTorch class and various functions to train a classifier on RoBERTa embeddings
References:
    - https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''

import torch
nn = torch.nn
import numpy as np
from typing import *
import csv
import argparse
from random_seed import load_random_seed

class NNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_fn = None) -> None:
        '''Initialize a one-layer classifier of embeddings
            - input_size: size of input embeddings
            - output_size: number of labels/classes
            - output_fn (optional): activation function to be used to get probabilities
        '''
        super(NNClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.output_fn = output_fn

    def forward(self, sentence_embeddings):
        '''Peform a forward pass on (a) batch of embedding(s)'''
        activation = nn.ReLU()
        intermediate = activation(self.hidden(sentence_embeddings))
        if self.output_fn:
            output = self.output_fn(self.classifier(intermediate))
        else:
            output = self.classifier(intermediate)
        return output

class SentenceData(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id], self.labels[id]

def train(model: NNClassifier, embeddings: List[str], labels: List[str], 
        random_seeds: List[int], batch_size: int, loss_fn: nn.modules.loss._Loss, 
        optimizer: torch.optim, epochs: int
        ) -> NNClassifier:
    '''Train the classifier on training data in batch and return the model'''

    # loop over epochs
    for i in range(epochs):

        # batch the data
        batched_data = batch_data(embeddings, labels, i, batch_size, random_seeds)

        # loop over the batched data
        for batch, (X, y) in enumerate(batched_data):

            # get the model predictions
            y_hats = model.forward(X)

            # initialize the gradient calculation
            optimizer.zero_grad()

            # print(torch.argmax(y_hats, dim=1), torch.argmax(y, dim=1))

            # calculate the loss
            loss = loss_fn(y_hats, y)

            # perform backpropogation
            loss.backward()
            optimizer.step()

            if batch % 64 == 0:
                loss = loss.item()
                current = (batch * len(X)) + (len(X) * epoch * len(batched_data))
                total = len(X) * epochs * len(batched_data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")
                correct = (torch.argmax(y_hats, dim=1)==torch.argmax(y, dim=1)).type(torch.float).sum().item()
                print(f"\tcorrect: {correct}/{len(X)}")

    return model


def batch_data(embeddings: List[np.array], labels: List[int], epoch: int,
        batch_size: int, random_seeds: List[int]) -> torch.utils.data.DataLoader:
    '''Combine combine labels and embeddings into a single list to feed to train''' 
    if isinstance(random_seeds, list):
        '''Use random labels'''
        dataset = (embeddings, labels)
        random_seed = random_seeds[epoch+1]
        np.random.seed(random_seed)
        np.random.shuffle(dataset)
        new_dataset = SentenceData(dataset[0], dataset[1])
        batched_data = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size)
        return batched_data
    else:
        new_dataset = SentenceData(embeddings, labels)
        batched_data = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
        return batched_data


def load_embeddings(file: str):
    '''Load pretrained embeddings from a file.'''
    embeddings = torch.load(file)
    return embeddings


def argparser():
    '''Parse the input arguments'''
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--raw_data', help='path to input raw data file')
    parser.add_argument('--input_embeddings', help="path to sentence embeddings file")
    parser.add_argument('--hidden_layer', help="size of the hidden layer of the classifier", type=int)
    parser.add_argument('--learning_rate', help="(float) learning rate of the classifier", type=float)
    parser.add_argument('--batch_size', help="(int) batch size of mini-batches for training", type=int)
    parser.add_argument('--epochs', help="(int) number of epochs for training", type=int)
    parser.add_argument('--random_seeds', help="(file) txt file of random seeds", default='None')
    args = parser.parse_args()
    return args

def main():

    # get input arguments
    args = argparser()

    # load embeddings
    embeddings = load_embeddings(args.input_embeddings)

    # load sentences and labels from csv
    sentences, labels = [], []
    with open(args.raw_data, 'r') as datafile:
        data = csv.reader(datafile)
        for row in data:
            sentences.append(row[0])
            labels.append(int(row[1]))

    # get model parameters from the data
    embedding_size = embeddings[0].size()[0]
    no_labels = len(set(labels))

    # convert labels to a tensor
    class_label = np.zeros((len(labels), no_labels), dtype=np.float32)
    for i, label in enumerate(labels):
        class_label[i, int(label)] = 1
    labels = torch.tensor(class_label)

    # initialize the classifier and optimizer
    classifier = NNClassifier(embedding_size, args.hidden_layer, no_labels, output_fn=nn.Sigmoid())
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCELoss()

    # get random seeds (optional)
    if args.random_seeds != "None":
        random_seeds = load_random_seed(args.epochs)
        torch.manual_seed(random_seeds[0])
    else:
        random_seeds = None

    # train the classifier
    classifier = train(
        classifier, 
        embeddings, 
        labels, 
        random_seeds, 
        args.batch_size, 
        loss_fn, 
        optimizer, 
        args.epochs)

if __name__ == '__main__':
    main()
