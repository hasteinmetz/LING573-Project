''' Alternatives to RNN class in rnn.py (Hilly)
'''

class RNN(nn.Module):
    def __init__(self, input_layer_size: int, hidden_layer_size: int, 
    loss_fn: _LossFunction, output_layer_size: int) -> None:
        '''Create a two-layer RNN model and define it's forward function'''
        super(CustomRNN, self).__init__()

        # set layer sizes
        self.hidden_size = input_layer_size
        self.input_size = hidden_layer_size
        self.output_size = output_layer_size

        # functions
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_fn = loss_fn
        self.activation = 'tanh'

        # define the layers and their sizes
        self.rnn_layer = nn.RNN(input_layer_size, hidden_layer_size, num_layers=1)

    def forward(self, input_vector: torch_tensor, hidden: torch_tensor):
        print(f"before: {input_vector.size()}") 
        output, hidden = self.rnn_layer(input_vector)
        output = self.softmax(output)
        return output, hidden
