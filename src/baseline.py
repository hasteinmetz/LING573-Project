#!/usr/bin/env python

from tkinter import W
import torch
import utils
import argparse
import numpy as np
from typing import *
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score, f1_score
from classifier_layer import NNClassifier, batch_data, load_embeddings, load_random_seed, train

nn = torch.nn

def eval_metrics(y_true, y_pred) -> None:
	#accuracy
	accuracy = accuracy_score(y_true, y_pred)
	print('accuracy: ', str(accuracy))
	#f1 score
	f1 = f1_score(y_true, y_pred)
	print('f1 score: ', str(f1))
	


def get_labels_tensor(labels: List[int], num_labels: int):
	class_labels = np.zeros((len(labels), num_labels), dtype=np.float32)
	for i, label in enumerate(labels):
		class_labels[i, int(label)] = 1
	return torch.tensor(class_labels)


def main(args: argparse.Namespace):
	print("parsing data...")
	train_sentences, train_labels = utils.read_data_from_file(args.train_sentences)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_sentences)
	
	# initialize roberta models
	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	model = RobertaModel.from_pretrained("roberta-base")

	# change the dimensions of the input sentences for debugging only
	# np.random.shuffle(train_sentences)
	# np.random.shuffle(train_labels)
	# train_sentences, train_labels = train_sentences[0:100], train_labels[0:100]

	training_inputs = tokenizer(train_sentences, return_tensors="pt", padding=True)
	print("running inputs through RoBERTA Base to generate embeddings...")
	with torch.no_grad():
		train_output = model(**training_inputs)
	
	# shape (batch_size, input_len, 768)
	last_hidden_states = train_output.last_hidden_state
	train_embeddings = torch.mean(last_hidden_states, dim=1).squeeze()
	embedding_size = train_embeddings[0].size()[0]
	num_labels = len(set(train_labels))

	# intialize classifier layer
	classifier_layer = NNClassifier(embedding_size, args.hidden_layer, num_labels, output_fn=nn.Sigmoid())
	optimizer = torch.optim.SGD(classifier_layer.parameters(), lr=args.learning_rate)
	loss_fn = nn.BCELoss()

    # get random seeds (optional)
	if args.random_seeds != "None":
		random_seeds = load_random_seed(args.epochs)
		torch.manual_seed(random_seeds[0])
	else:
		random_seeds = None
	
	# set up classifier layer
	print(train_labels)
	print("setting up and training classifier layer...")
	train_labels_tensor = get_labels_tensor(train_labels, num_labels)
	classifier = train(
		classifier_layer, 
		train_embeddings, 
		train_labels_tensor, 
		args.random_seeds,
		args.batch_size, 
		loss_fn, 
		optimizer, 
		args.epochs)

	# get embeddings
	print("running inputs through RoBERTA Base to generate embeddings...")
	dev_inputs = tokenizer(dev_sentences, return_tensors="pt", padding=True)
	with torch.no_grad():
		dev_output = model(**dev_inputs)
	last_hidden_states = dev_output.last_hidden_state
	dev_embeddings = torch.mean(last_hidden_states, dim=1).squeeze()
	dev_labels_tensor = get_labels_tensor(dev_labels, num_labels)
	dev_batched_data = batch_data(dev_embeddings, dev_labels_tensor, epoch=0, 
		batch_size=args.batch_size, random_seeds=args.random_seeds)

	print("running model prediction...")

	Y = []
	with torch.no_grad():
		for X, y in dev_batched_data:
			pred_label = classifier(X)
			# Y is a list of tensor
			Y.append(torch.argmax(pred_label, dim=1))

	predicted_labels = torch.cat(Y)

	y_pred = predicted_labels.numpy()
	eval_metrics(dev_labels, y_pred)

	#write results to output file
	utils.write_output_to_file(args.output_file, dev_sentences, y_pred, encoding='utf-8')
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_sentences', help="path to input training data file")
	parser.add_argument('--dev_sentences', help="path to input dev data file")
	parser.add_argument('--hidden_layer', help="size of the hidden layer of the classifier", type=int)
	parser.add_argument('--learning_rate', help="(float) learning rate of the classifier", type=float)
	parser.add_argument('--batch_size', help="(int) batch size of mini-batches for training", type=int)
	parser.add_argument('--epochs', help="(int) number of epochs for training", type=int)
	parser.add_argument('--random_seeds', help="(file) txt file of random seeds", default='None')
	parser.add_argument('--output_file', help="path to output data file")
	args = parser.parse_args()

	main(args)
