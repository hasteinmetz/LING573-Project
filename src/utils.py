"""
script for common functions
"""
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

# TODO: is str the right type for label?
def read_data_from_file(file_path: str, seperator: str = ',', encoding: str = 'utf-8') -> Tuple[List[str], List[str]]:
	'''
	arguments:
		- file_path: full file path pointing to input file, expects two columns with following schema (Text, Label)
		- separator: specifies which char is used to separate columns in input file
		- encoding: encoding of input file

	parses given file, returning a tuple with two lists, one representing the input data text and the other representing output data labels
	'''

	data, labels = [], []
	with open(file_path) as f:
		for line in f:
			contents = line.strip().split(seperator)
			data.append(contents[0])
			labels.append(contents[1])
	
	return data, labels