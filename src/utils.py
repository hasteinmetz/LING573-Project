#!/usr/bin/env python

'''
script for common functions
'''

import csv
import json
import time
import spacy
import re
import numpy as np
import pandas as pd
from typing import *
from functools import reduce
from sklearn.preprocessing import Normalizer

def read_file(file_path: str, seperator: str = ',', encoding: str = 'utf-8') -> List[List[str]]:
	'''
	arguments:
		- file_path: full file path pointing to input file
		- separator: specifies which char is used to separate columns in input file
		- encoding: encoding of input file

	parses given file, returning a list of a list of strings representing each row in the input file
	'''
	data = []
	with open(file_path) as f:
		for line in f:
			contents = line.strip().split(seperator)
			data.append(contents)
	
	return data


def read_data_from_file(filepath: str, encoding: str = 'utf-8', index: int = 1) -> Tuple[List[str], np.ndarray]:
	'''
	arguments:
		- file_path: full file path pointing to input file, expects two columns with following schema (Text, Label)
		- separator: specifies which char is used to separate columns in input file
		- encoding: encoding of input file

	parses given file, returning a tuple with two lists, one representing the input data text and the other representing output data labels
	'''
	sentences, labels = [], []
	with open(filepath, 'r', encoding=encoding) as datafile:
		data = csv.reader(datafile, delimiter=',', quotechar='"')
		for row in data:
			if isanumber(row[index]):
				sentences.append(row[0])
				labels.append(row[index])
	return sentences, np.asarray(labels, dtype=np.float32)

def isanumber(string) -> bool:
	'''Check is a string is a float or integer'''
	if re.search(r'[0-9]+[\.]?[0-9]+', string) or string.isnumeric():
		return True
	else:
		return False

def write_output_to_file(filepath: str, data: List[str], labels: np.ndarray, encoding: str = 'utf-8') -> None:
	with open(filepath, "w", newline='', encoding="utf-8") as my_csv:  # create training data file
		my_writer = csv.writer(my_csv)
		for i in range(len(data)):
			my_writer.writerow([data[i], labels[i]])
	my_csv.close()


def load_json_config(filepath: str) -> dict:
	'''
	arguments:
		- filepath: full filepath pointing to json config file

	opens file and loads configuration into a dictionary
	'''
	config = None
	with open(filepath, 'r') as f:
		config = json.load(f)
	
	return config


def normalize_vector(*arrays: np.ndarray) -> np.ndarray:
	'''Take several arrays and concatenate them column-wise before normalizing each row'''
	concatenated = np.concatenate(arrays, axis=1)
	norm = Normalizer()
	return norm.fit(concatenated).transform(concatenated)


def lemmatize(sentences: List[str]) -> List[str]:
	'''Process the sentence into a string of lemmas (to potentially improve the TF-IDF measure)
	This function requires spacy to use'''
	processer = spacy.load("en_core_web_sm")
	lemmatizer = processer.get_pipe("lemmatizer")
	to_str = lambda x, y: x + " " + y
	lemmatize = lambda x: reduce(to_str, [tkn.lemma_ for tkn in processer(x)])
	new_sents = [lemmatize(x) for x in sentences]
	return new_sents


def get_time(start_time: float) -> str:
	minutes, sec = divmod(time.time() - start_time, 60)
	return f"{str(round(minutes))}min {str(round(sec))}sec"


def read_from_tsv(lex_data: str) -> Tuple[Dict[str,str], set]:
	"""
	read in and output hurtlex dictionary in the format of 'lemma:[category, tag]'
	"""
	output_dict = {}
	
	df = pd.read_csv(lex_data, sep='\t', header=0, usecols=[1,2,4])
	feature_list = set(df['category'].tolist())
	
	df['category'] = df[['category', 'pos']].values.tolist()
	output_dict = dict(zip(df.lemma, df.category))
	
	return output_dict, feature_list