# FOLLOWING RNN TUTORIAL: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from torch import cat, zeros, tensor
from torch import nn
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

# class Transformer(nn.Transformer):
#     '''Class to define a BERT-style transformer.
#     See: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
#     Parameter defaults are based on BERT tiny (https://github.com/google-research/bert)'''
#     def __init__(self, 
#             d_model: int = 128, 
#             nhead: int = 2, 
#             num_encoder_layers: int = 6, 
#             num_decoder_layers: int = 6, 
#             dim_feedforward: int = 2048, 
#             dropout: float = 0.1, 
#             activation: Optional[Any] = "relu", 
#             custom_encoder: Optional[Any] = None, 
#             custom_decoder: Optional[Any] = None, 
#             layer_norm_eps: float = 0.00001, 
#             batch_first: bool = False, 
#             norm_first: bool = False, 
#             device=None, 
#             dtype=None) -> None:
#         super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, 
#             dim_feedforward, dropout, activation, custom_encoder, custom_decoder, 
#             layer_norm_eps, batch_first, norm_first, device, dtype)

# def train_transformer():
#     return

class RNN(nn.Module):
    def __init__(self, embeddings: dict, hidden_layer_size, 
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

    def forward(self, input: str, hidden):
        input_vector = self.embeddings[input]
        combined_vector = cat((input_vector, hidden), 1)
        hidden = self.input_to_hidden(combined_vector)
        output = self.input_to_output(combined_vector)
        output = self.softmax(output)
        return output, hidden

    def init_hidden_layer(self):
        return zeros(1, self.hidden_size)

# want to input a tensor of size: word_embedding x sentence size

def train_rnn(model: RNN, input_tensor, true_label):
    '''Instructions on training and updating a model'''
    # initialize the hidden layer
    hidden = model.init_hidden_layer()

    # initialize the gradient calculation
    model.zero_grad()

    # do you want to do like a "for batch" or something?
    # TODO: CREATE TENSORS OF THE BATCHES OF DATA

    # get the result for the input tensor
    for i in range(0, input_tensor[0].size()):
        output, hidden = model.forward(input_tensor[i], hidden)

    loss_function = nn.NLLLoss()

    loss = loss_function(output, true_label)

    loss.backwards()

    model.optimizer.step()

    return output, loss.item()

def training_iterations(model: RNN, dataset: List[str],
    epochs: int = 1, learning_rate: int = 0.005):
    iterations = len(dataset) * epochs
    for 
