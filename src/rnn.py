# FOLLOWING RNN TUTORIAL: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from torch import cat, zeros, tensor
from torch import nn
from torch.utils import DataLoader
import numpy as np

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
    def __init__(self, embeddings: dict, hidden_layer_size, loss_fn,
            input_size, output_size, optimizer, learning_rate) -> None:
        '''Create a two-layer RNN model and define it's forward function'''
        super(RNN).__init__()
        self.embeddings = embeddings
        self.hidden_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_to_hidden = nn.Linear(input_size + hidden_layer_size, 
            hidden_layer_size)
        self.input_to_output = nn.Linear(input_size + hidden_layer_size, 
            output_size)
        '''Set the output layer function (if 2 use sigmoid)'''
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn()

    def forward(self, input_vector, hidden):
        combined_vector = cat((input_vector, hidden), 1)
        hidden = self.input_to_hidden(combined_vector)
        output = self.input_to_output(combined_vector)
        output = self.softmax(output)
        return output, hidden

    def init_hidden_layer(self):
        return zeros(1, self.hidden_size)

# want to input a tensor of size: word_embedding x sentence size

def train_rnn(model: RNN, batch, true_labels):
    '''Instructions on training and updating a model'''
    # initialize the hidden layer
    hidden = model.init_hidden_layer()

    # initialize the gradient calculation
    model.zero_grad()

    # get the result for the input tensor
    model_outputs = []
    for input_tensor in batch:
        for i in range(0, input_tensor[0].size()):
            output, hidden = model.forward(input_tensor[i], hidden)
        model_outputs.append(output)

    loss = model.loss_fn(model_outputs, true_labels)

    loss.backwards()

    model.optimizer.step()

    return model_outputs, loss.item()

def training_iterations(model: RNN, dataset: List[str],
    epochs: int = 1, batch_size: int = 1, learning_rate: int = 0.005) -> None:
    iterations = len(dataset) * epochs
    data = DataLoader(dataset, batch_size)
    for batch, (X, y) in enumerate(data):
        output, loss = train_rnn(model, X, y)
    
        # print progress to stdout
        if batch % 50 == 0:
            loss, current_iter = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current_iter:>5d}/{batch_size:>5d}]")
            correct = (output.argmax(1)==y).type(torch.float).sum().item()
            print(f"correct: {correct}"

def main():
    
    