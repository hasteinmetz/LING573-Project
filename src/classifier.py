import torch
nn = torch.nn

class NNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_fn = nn.ReLU(), activation_fn=nn.Softmax()) -> None:
        '''Initialize a one-layer classifier of embeddings
            - input_size: size of input embeddings
            - output_size: number of labels/classes
            - output_fn (optional): activation function to be used to get probabilities
        '''
        super(NNClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.output_fn = output_fn
        self.activation_fn = activation_fn

    def forward(self, sentence_embeddings):
        '''Peform a forward pass on (a) batch of embedding(s)'''
        intermediate = self.activation_fn(self.hidden(sentence_embeddings))
        output = self.output_fn(self.classifier(intermediate))
        return output