# FOLLOWING RNN TUTORIAL: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from torch import (
    cat as torch_cat, 
    zeros as torch_zeros, 
    tensor as torch_tensor, 
    float as torch_float,
    empty as torch_empty,
    no_grad
)
from torch import nn, optim
from torch.nn.modules.loss import _Loss as _LossFunction # to include types in the function headers
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import typing
import re
from nltk.tokenize import word_tokenize

class RNN(nn.Module):
    def __init__(self, hidden_layer_size: int, loss_fn: _LossFunction, 
    input_layer_size: int, output_layer_size: int) -> None:
        '''Create a two-layer RNN model and define it's forward function'''
        super(RNN, self).__init__()

        # set layer sizes
        self.hidden_size = hidden_layer_size
        self.input_size = input_layer_size
        self.output_size = output_layer_size

        # functions
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_fn = loss_fn
        self.activation = nn.Tanh()

        # define the layers and their sizes
        self.input_to_hidden = nn.Linear(self.input_size + self.hidden_size,
            self.hidden_size)
        self.input_to_output = nn.Linear(self.input_size + self.hidden_size, 
            self.output_size)

    def forward(self, input_vector: torch_tensor, hidden: torch_tensor):
        # debug: print(f"before: {input_vector.size()}") 
        input_vector.resize_(1, input_vector.size()[0]) # resize the input vector
        # debug: print(f"after: {input_vector.size()}")
        combined_vector = torch_cat((input_vector, hidden), -1)
        # debug: print(f"combined: {combined_vector.size()}")
        hidden = self.activation(self.input_to_hidden(combined_vector))
        output = self.softmax(self.input_to_output(combined_vector))
        return output, hidden

    def init_hidden_layer(self):
        return torch_zeros(1, self.hidden_size)

class SentenceData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id], self.labels[id]


def train_rnn(model: RNN, batch: torch_tensor, true_labels: list[int], optimizer: optim):
    '''Instructions on training and updating a model'''
    
    # initialize the gradient calculation
    optimizer.zero_grad()

    # get the result for the input tensor
    model_outputs = torch_empty((batch.size()[0], model.output_size))

    # get the outputs for a batch
    for j, input_tensor in enumerate(batch):
        # initialize the hidden layer
        hidden = model.init_hidden_layer()
        for i in range(0, input_tensor.size()[0]):
            output, hidden = model(input_tensor[i], hidden)
        model_outputs[j,:] = output

    loss = model.loss_fn(model_outputs, true_labels)
    # uncomment the line below to help debug:
    # print(loss, model_outputs, true_labels)

    # perform backpropogation
    loss.backward()
    optimizer.step()

    return model_outputs, loss


def my_collate(batch, max_length):
    batch = [torch_cat((x, torch_zeros((x.size()[], max_length-x.size()[]))), 1) for x in batch]

def batch_data(embeddings: list[np.array], labels: list[int], 
        batch_size: int) -> DataLoader:
    '''Combine combine labels and embeddings into a single list to feed to train''' 
    # convert labels to a tensor
    class_labels = []
    for label in labels:
        zeros = [0] * len(set(labels))
        zeros[int(label)] = 1
        class_labels.append(torch_tensor(zeros))
    labels = class_labels
    new_dataset = SentenceData(embeddings, labels)
    batched_data = DataLoader(new_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    return batched_data


def training_iterations(model: RNN, sentences, labels, optimizer,
    epochs: int = 2, batch_size: int = 50, print_every: int = 1) -> None:
    data = batch_data(sentences, labels, batch_size)
    for i in range(epochs):
        for batch, (X, y) in enumerate(data):
            output, loss = train_rnn(model, X, y, optimizer)
            # print progress to stdout
            if batch % print_every == 0:
                current_iter = (i * len(data)) + batch + 1
                print(f"loss: {loss:>7f} [{current_iter:>5d}/{len(data) * epochs:>5d}]")
                correct = (output.argmax(1)==y).type(torch_float).sum().item()
                print(f"correct: {correct}/{batch_size}")


def create_and_train_rnn(
        sentences_as_embeddings: list[np.array],
        labels: list[int],
        hidden_size: int,
        loss_function: _LossFunction,
        optimizer: optim,
        learning_rate: float,
        epochs: int
    ):
    '''Function to load parameters and train a single layered RNN model.
        Args:
            sentences_as_embeddings: a list of training sentences that have 
                converted to a list of word embeddings
            labels: the output labels of the sentences (e.g., 1 for humorous
                and 0 for not humorous)
            loss_function: the torch.optim loss function to be used by the rnn
            hidden_size: the size of the RNN's hidden layer
            epochs: the number of epochs to train the RNN's layers
    '''
    no_classes = len(set(labels))
    input_vector_dim = sentences_as_embeddings[0].shape[1] # size of the embeddings (taken from first sentence)
    # debug: print(f'input_vector: {input_vector_dim}')
    rnn = RNN(input_vector_dim, hidden_size, loss_function, no_classes)
    opt = optimizer(rnn.parameters(), learning_rate)
    training_iterations(rnn, sentences_as_embeddings, labels, opt, epochs)
    return rnn

def test_on_imdb():
    from nltk.corpus import movie_reviews
    positive = movie_reviews.fileids('pos')
    negative = movie_reviews.fileids('neg')
    training = [(filter(lambda x: re.search(r'\w', x), movie_reviews.words(f)), 1) for f in positive]
    training.extend([(filter(lambda x: re.search(r'\w', x), movie_reviews.words(f)), 0) for f in negative])
    embeddings = load_embeddings('src/data/glove.6B.50d.txt')
    sentences, labels = [], []
    for sentence, label in training:
        vectors = []
        for word in sentence:
            if word in embeddings:
                val = embeddings[word]
            else:
                val = np.zeros(len(embeddings['a']))
            vectors.append(val)
        sentences.append(torch_tensor(np.asarray(vectors, dtype=np.float32)))
        labels.append(label)
    rnn = create_and_train_rnn(sentences, labels, 100, nn.NLLLoss(), optim.SGD, 0.01, 5)

def load_embeddings(file: str):
    '''load pretrained embeddings from a .txt file. 
    The file should be split with spaces'''
    embeddings = {}
    with open(file, 'r') as glove:
        for l in glove:
            line = l.strip().split()
            token, vector = line[0], np.asarray(line[1:], dtype=np.float64)
            embeddings[token] = vector
    return embeddings

def debug():
    embeddings = load_embeddings('src/data/glove.6B.50d.txt')
    # create tensors for each sentence
    sentences, labels = [], []
    with open('src/data/hahackathon_prepo1_dev.csv', 'r') as jokes:
        reader = csv.reader(jokes, delimiter=',', quotechar='"')
        for row in reader:
            tokens = word_tokenize(row[0].lower())
            tokens = filter(lambda x: re.search(r'\w', x), tokens)
            sentence = []
            for word in tokens:
                if word in embeddings:
                    val = embeddings[word]
                else:
                    val = np.zeros(len(embeddings['a']))
                sentence.append(val)
            sentences.append(sentence)
            labels.append(int(row[1]))
    rnn = create_and_train_rnn(sentences, labels, 50, nn.NLLLoss(), optim.Adam, 0.0001, 100)

if __name__ == '__main__':
    test_on_imdb()
