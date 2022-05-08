import csv
import nltk
import numpy as np
from typing import *

def get_ner_matrix(input_sentences: List[str]) -> np.ndarray:  
	'''
	arguments:
		- input_sentences: list of input data to extract NER features for
	returns:
		a numpy array of [num_documents, 6] where 6 is the number of NER categories
	'''
	label_list = ["GPE", "PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "GSP"]  # hardcoded to ensure length
	label_dict = {
		"GPE": 0,
		"PERSON": 1,
		"ORGANIZATION": 2,
		"FACILITY": 3,
		"LOCATION": 4,
		"GSP": 5
	}
	total_list = []
	for row in input_sentences:
		tokenized = nltk.word_tokenize(row)  # tokenize
		tagged = nltk.pos_tag(tokenized)  # get pos-tags, necessary for nltk.ne_chunk
		ne_tree = nltk.ne_chunk(tagged)  # develop tree, which defines what is a named entity

		ne_list = []

		for item in ne_tree:
			if not isinstance(item, tuple):  # non-Named-entities are saved as tuples, others are Named entities
				ne_list.append(item)  # append Named entities

		zerod = np.zeros(len(label_list))
		for item in ne_list:
			label = item.label()
			index = label_dict.get(label)
			if index is not None:
				zerod[index] += 1
		total_list.append(zerod)

	ner_array = np.asarray(total_list)

	return ner_array
