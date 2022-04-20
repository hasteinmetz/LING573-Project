# FOLLOWING RNN TUTORIAL: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from torch import cat, zeros, tensor, float, empty
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import csv
import typing
import re
from nltk.tokenize import word_tokenize

def numpy_to_tensor(*args: np.ndarray):
    '''Take NumPy arrays to and return them as tensors'''
    return map(tensor, args)


def set_classes(categories):
    '''Create a dictionary of output values corresponding to their
    respective classes i.e. {humor:1, not_humor: 0}'''
    classes, values = []
    for i, c in enumerate(categories):
        classes.append(c)
        values.append(i)
    return classes, values

class RNN(nn.Module):
    def __init__(self, hidden_layer_size: int, loss_fn, input_size: int, output_size: int) -> None:
        '''Create a two-layer RNN model and define it's forward function'''
        super(RNN, self).__init__()
        self.hidden_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_to_hidden = nn.Linear(input_size + hidden_layer_size, hidden_layer_size)
        self.input_to_output = nn.Linear(input_size + hidden_layer_size, output_size)
        '''Set the output layer function (if 2 use sigmoid)'''
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_fn = loss_fn

    def forward(self, input_vector, hidden):
        # resize the input vector:
        input_vector.resize_(1, input_vector.size()[0])
        combined_vector = cat((input_vector, hidden), -1)
        hidden = self.input_to_hidden(combined_vector)
        output = self.input_to_output(combined_vector)
        output = self.softmax(output)
        return output, hidden

    def init_hidden_layer(self):
        return zeros(1, self.hidden_size)

# want to input a tensor of size: word_embedding x sentence size

def train_rnn(model: RNN, batch, true_labels, optimizer):
    '''Instructions on training and updating a model'''
    # initialize the hidden layer
    hidden = model.init_hidden_layer()

    # initialize the gradient calculation
    optimizer.zero_grad()

    # get the result for the input tensor
    model_outputs = empty((batch.size()[0], model.output_size))
    for j, input_tensor in enumerate(batch):
        for i in range(0, input_tensor.size()[0]):
            output, hidden = model.forward(input_tensor[i], hidden)
        model_outputs[j,:] = output

    loss = model.loss_fn(model_outputs, true_labels)
    # print(loss, model_outputs, true_labels)

    loss.backward()
    optimizer.step()

    return model_outputs, loss

def training_iterations(model: RNN, dataset: list[np.array], optimizer,
    epochs: int = 2, batch_size: int = 50) -> None:
    data_as_tensors, labels = [tensor(x[0]) for x in dataset], [y[1] for y in dataset]
    padded_data = nn.utils.rnn.pad_sequence(data_as_tensors, batch_first=True)
    dataset_new = [(x, y) for x, y in zip(padded_data, labels)]
    data = DataLoader(dataset_new, batch_size, drop_last=True, shuffle=True) 
    for i in range(epochs):
        for batch, (X, y) in enumerate(data):
            output, loss = train_rnn(model, X, y, optimizer)
            # print progress to stdout
            if batch % batch_size == 0:
                current_iter = batch + ((i+1) * len(data))
                print(
                    f"loss: {loss:>7f}" + 
                    f"[{current_iter:>5d}/{len(data) * epochs:>5d}]" +
                    f"(out of training set size {len(dataset)})")
                correct = (output.argmax(1)==y).type(float).sum().item()
                print(f"correct: {correct}/{batch_size}")

def debug():
   # load pretrained glove embeddings to test neural model and make sure it works
    embeddings = {}
    with open('src/data/glove.6B.50d.txt', 'r') as glove:
        for l in glove:
            line = l.strip().split()
            token, vector = line[0], np.asarray(line[1:], dtype=np.float64)
            embeddings[token] = vector
    
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
    dataset = [(np.asarray(x, dtype=np.float32), y) for x, y in zip(sentences, labels)]
    rnn = RNN(200, nn.NLLLoss(), 50, 2)
    optimizer = optim.Adam(rnn.parameters(), 0.0001)
    training_iterations(rnn, dataset, optimizer, epochs=100)

if __name__ == '__main__':
    debug()
