import torch
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

inputs = tokenizer("Why were the two homosexual melons protesting at city hall? Because they cantaloupe", return_tensors="pt")

# no_grad means don't perform gradient descent/backprop
# run when model is in eval mode, which it is by default when loading using pre-trained method
with torch.no_grad():
	outputs = model(**inputs)


# get vector representing last hidden state
# shape (batch_size, input_len, 768)
# batch_size = number of input sentences
# input_len depends on how tokenizer decides to prepare input sentence
# 768 is the dimension of the layer
last_hidden_states = outputs.last_hidden_state

# get sentence embedding
sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze()
