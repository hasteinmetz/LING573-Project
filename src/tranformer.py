# FOLLOWING RNN TUTORIAL: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from torch import cat, zeros, tensor
from torch import nn
import torch.functional as nn_functions
import numpy as np

def numpy_to_tensor(*args):
    '''Take NumPy arrays to and return them as tensors'''
    return map(tensor, args)

class RNN(nn.Module):
    def __init__(self, hidden_layer_size, input_size, output_size=2) -> None:
        '''Create a two-layer RNN model and define it's forward function'''
        super(RNN).__init__()
        self.hidden_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_to_hidden = nn.Linear(input_size + hidden_layer_size, 
            hidden_layer_size)
        self.input_to_output = nn.Linear(input_size + hidden_layer_size, 
            output_size)
        '''Set the output layer function (if 2 use sigmoid)'''
        self.softmax = nn_functions.log_softmax(dim=1)

    def forward(self, input, hidden):
        combined_vector = cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined_vector)
        output = self.input_to_output(combined_vector)
        output = self.softmax(output)
        return output, hidden

    def init_hidden_layer(self):
        return zeros(1, self.hidden_size)

def set_classes(categories):
    '''Create a dictionary of output values corresponding to their
    respective classes i.e. {humor:1, not_humor: 0}'''
    classes, values = []
    for i, c in enumerate(categories):
        classes.append(c)
        values.append(i)
    return classes, values

def train_rnn(model: RNN, input_tensor):
    hidden_tensor = model.init_hidden_layer()
    # how does this work? 