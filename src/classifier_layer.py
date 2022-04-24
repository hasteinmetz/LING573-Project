'''
PyTorch class and various functions to train a classifier on RoBERTa embeddings
References:
    - https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''

import torch
nn = torch.nn
import numpy as np
from typing import *
import re
from functools import reduce
import csv
import argparse

class NNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_fn = None) -> None:
        '''Initialize a one-layer classifier of embeddings
            - input_size: size of input embeddings
            - output_size: number of labels/classes
            - output_fn (optional): activation function to be used to get probabilities
        '''
        super(NNClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        # TODO: Find a random seed for these layers
        self.hidden.weight.data.fill_(0.0001)
        self.hidden.bias.data.fill_(0.0001)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.classifier.weight.data.fill_(0.0001)
        self.classifier.bias.data.fill_(0.0001)
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

def train(model: NNClassifier, batched_data: torch.utils.data.DataLoader, 
        loss_fn: nn.modules.loss._Loss, optimizer: torch.optim, epochs: int
        ) -> NNClassifier:
    '''Train the classifier on training data in batch and return the model'''
    # loop over epochs
    for i in range(epochs):
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
                current = (batch * len(X)) + (len(X) * i * len(batched_data))
                total = len(X) * epochs * len(batched_data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{total:>5d}]")
                correct = (torch.argmax(y_hats, dim=1)==torch.argmax(y, dim=1)).type(torch.float).sum().item()
                print(f"\tcorrect: {correct}/{len(X)}")

    return model


def batch_data(embeddings: list[np.array], labels: list[int], 
        batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    '''Combine combine labels and embeddings into a single list to feed to train''' 
    new_dataset = SentenceData(embeddings, labels)
    # TODO: Find a way to get a random seed for this shuffle
    batched_data = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle)
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

    # batch the data
    batched_data = batch_data(embeddings, labels, args.batch_size)

    # train the classifier
    classifier = train(classifier, batched_data, loss_fn, optimizer, args.epochs)

def debug():
    '''Make sure the classifier works using movie review data'''

    # load NLTK data
    from nltk.corpus import movie_reviews
    positive = movie_reviews.fileids('pos')
    negative = movie_reviews.fileids('neg')

    # filter punctuations in movie reviews
    training = [(filter(lambda x: re.search(r'\w', x), movie_reviews.words(f)), 1.) for f in positive]
    training.extend([(filter(lambda x: re.search(r'\w', x), movie_reviews.words(f)), 0.) for f in negative])
    
    # load embedddings and change words to their embeddings
    embeddings = load_embeddings('src/data/glove.6B.50d.txt')
    sentences, labels = [], []
    embedding_size, no_labels = len(embeddings['a']), 2
    for sentence, label in training:
        vectors = []
        for i, word in enumerate(sentence):
            if word in embeddings:
                val = embeddings[word]
            else:
                val = np.zeros(embedding_size, np.float32)
            vectors.append(val)
        # get the average vector of the document
        centroid_vector = reduce(lambda x,y: x+y, vectors)/i
        sentences.append(centroid_vector)
        labels.append(label)

    # initialize the classifier, batch the data, and train the classifier
    classifier = NNClassifier(embedding_size, no_labels, nn.Softmax())
    batched_data = batch_data(sentences, labels, 64)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    train(classifier, batched_data, loss_fn, optimizer, 1000)

    size = len(batched_data.dataset)
    num_batches = len(batched_data)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in batched_data:
            pred = classifier(X)
            y = y.long()
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    main()