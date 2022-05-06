#!/usr/bin/env python

"""
script for common functions
"""
import csv
import numpy as np
from typing import *

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

def read_data_from_file(filepath: str, encoding: str = 'utf-8') -> Tuple[List[str], np.ndarray]:
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
			sentences.append(row[0])
			labels.append(int(row[1]))
	return sentences, np.asarray(labels, dtype=int)

def write_output_to_file(filepath: str, data: List[str], labels: np.ndarray, encoding: str = 'utf-8') -> None:
	with open(filepath, "w", newline='', encoding="utf-8") as my_csv:  # create training data file
		my_writer = csv.writer(my_csv)
		for i in range(len(data)):
			my_writer.writerow([data[i], labels[i]])
	my_csv.close()
