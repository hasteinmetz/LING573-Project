import torch
from typing import *
import argparse
from transformers import RobertaTokenizer, RobertaModel
import utils


def get_embeddings(data: List[str], output_path: str) -> None:
	print("initializing models to get embeddings...")
	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	model = RobertaModel.from_pretrained("roberta-base")

	inputs = tokenizer(data, return_tensors="pt", padding=True)

	print("running inputs through RoBERTA Base to generate embeddings...")
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
	sentence_embeddings = torch.mean(last_hidden_states, dim=1).squeeze()

	print("saving embeddings to file at {}".format(output_path))
	#write embedding to output file
	torch.save(sentence_embeddings, output_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_sentences', help="path to input data file")
	parser.add_argument('--output_file', help="path to output data file")
	args = parser.parse_args()

	input_sentences = None
	print("parsing input data...")
	if args.input_sentences[-4:] == ".csv":
		input_sentences, _ = utils.read_data_from_file(args.input_sentences)
	elif args.input_sentences[-4:] == ".tsv":
		input_sentences, _ = utils.read_data_from_file(args.input_sentences, separator='\t')
	else:
		print("unsupported file format {} . Please use either a .tsv or a .csv for the input data")
	
	if input_sentences != None:
		get_embeddings(input_sentences, args.output_file)
