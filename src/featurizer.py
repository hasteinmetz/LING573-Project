import nltk
import spacy
import utils
import numpy as np
import pandas as pd
from typing import *
from empath import Empath
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFGenerator:
	# TODO: look into DELTA TF-IDF https://ebiquity.umbc.edu/_file_directory_/papers/446.pdf
	'''Class to store a fitted TD-IDF generator and sentences in order to keep the 
	fitting consistent'''
	def __init__(self, sentences: List[str], stop_words: str, concat_labels: List[str]):
		self.sentences = sentences
		self.vectorizer = get_vocabulary(sentences, stop_words, concat_labels)

	def get_tfidf(self, sentences: List[str]) -> np.ndarray:
		'''Get the TF-IDF of the sentences using a TfidfVectorizer fitted to the training data'''
		matrix = self.vectorizer.transform(sentences)
		return matrix.toarray()

	def reinitialize_matrix(self, stop_words, concat_labels):
		self.vectorizer = get_vocabulary(self.sentences, stop_words, concat_labels)

def get_ner_matrix(data: List[str]) -> np.ndarray:  
	'''
	arguments:
		- data: list of input data to extract NER features for
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
	for row in data:
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


def create_lexical_matrix(sentences: List[str], chars: List[str]) -> np.ndarray:
	'''Take a dataset and return an [num_sentences, num_chars] matrix of counts
	of the number of times each character appears in a list'''
	# initialize a list to store vectors
	return_vectors = []
	for s in sentences:
		char_counts = {c : s.count(c) for c in chars}
		return_vectors.append(char_counts)
	df = pd.DataFrame(return_vectors)
	return df.to_numpy()


def get_empath_ratings(sentences: List[str]) -> np.ndarray:
	'''Get EMPATH (https://github.com/Ejhfast/empath-client) ratings for sentences'''
	lexicon = Empath()
	dictionary = {k: [] for k in lexicon.cats.keys()}
	for s in sentences:
		analyzed_s = lexicon.analyze(s, normalize=True)
		for k, v in analyzed_s.items():
			dictionary[k].append(v)
	as_lists = [dictionary[k] for k in dictionary]
	return np.column_stack(as_lists)


def get_vocabulary(training_sents: List[str], stop_words: str = None, 
	concat_labels: List[str] = None) -> TfidfVectorizer:
	'''Get counts of the training data set to calculate TD-IDF
		Args:
			- training_sents: sentences from the training data
			- stop_words: whether or not to include stop words in the vectorizer
				To exclude stop words use the string 'english' as a parameter
			- concat_labels: whether to concatenate the each label's sentences into a single document
				- If None don't concatenate
				- Otherwise provide a list of labels with corresponding indices to the sentences
	''' 
	vectorizer = TfidfVectorizer(decode_error='ignore', stop_words=stop_words) # ALL WORDS ARE LOWERCASED!
	if isinstance(concat_labels, list):
		sents = {k: '' for k in set(concat_labels)}
		for s, label in zip(training_sents, concat_labels):
			sents[label] += s
		sents = [sents[1], sents[0]]
		fitted_vectorizer = vectorizer.fit(sents) # NOTE: DO NOT fit the vectorizer to the test data!
	else:
		fitted_vectorizer = vectorizer.fit(training_sents) # NOTE: DO NOT fit the vectorizer to the test data!

	return fitted_vectorizer


def get_tfidf(sentences: List[str], fitted_vectorizer: TfidfVectorizer) -> np.ndarray:
	'''Get the TF-IDF of the sentences using a TfidfVectorizer fitted to the training data'''
	matrix = fitted_vectorizer.transform(sentences)
	return matrix.toarray()


def check_phrase(sentence: str, lex_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
	'''
	arguments:
		- sentence: lemmatized input data sentence
		- lex_dixt: hurtlex dictionary

	check phrases in hurtlex
	'''
	contain_phrase = False
	tokens = []

	for k, v in lex_dict.items():
		if k in sentence and len(k.split()) > 1:
			contain_phrase = True
			tokens.append(v[0])

	return contain_phrase, tokens


def count_feature(sentence: str, lex_dict: Dict[str, str], feature_list: set, tagger) -> np.ndarray:
	'''
	arguments:
		- sentence: input sentence
		- lex_dict: hurtlex lexicon dictionary
		- feature_list: hurtlex category list
		- tagger: pos tagger
	returns:
		a (17) vector representing hurtlex semantic space of input sentence
	'''
	#set up
	count = dict.fromkeys(feature_list, 0)
	spacy_s = tagger(sentence)

	#check hurtlex phrases in the lemmatized sentence
	test = ' '.join([token.lemma_ for token in spacy_s])
	cond, cats = check_phrase(test, lex_dict)

	if cond:
		for tag in cats:
			count[tag] += 1
	else:
		for token in spacy_s:
			if token.lemma_ in lex_dict:
				# check if pos_tag matches
				# if so, add a count
				if token.tag_.lower()[0] == lex_dict[token.lemma_][1] or (token.tag_[0] == 'j' and lex_dict[token.lemma_][1] == 'a'):
					count[lex_dict[token.lemma_][0]] += 1

	feature = []
	for k, v in sorted(count.items()):
		feature.append(v)

	return feature


def extract_hurtlex(sentences: List[str], lex_dict: Dict[str, str], feature: set) -> np.ndarray:
	'''
	arguments:
		- sentences: list of input data to be featurized
		- lex_dict: hurtlex lexicon dictionary 
		- feature: hurtlex category list
	returns:
		a (n_samples, 17) vector representing the hurtlex semantic space of each sentence
	'''
	#set up
	tagger = spacy.load("en_core_web_sm")

	features = []
	for data in sentences:
		s = count_feature(data, lex_dict, feature, tagger)
		features.append(s)

	return np.array(features)


def featurize(sentences: List[str], labels: np.ndarray, hurtlex_dict: Dict[str, str], hurtlex_cat: set) -> np.ndarray:
# def featurize(sentences: List[str], tfidf_gen: TFIDFGenerator) -> np.ndarray:
	'''
	arguments:
		- sentences: list of input data to be featurized
		- labels: corresponding labels for sentenceds
	returns:
		a (n_samples, vector_space_size) feature vector 
	
	featurizes the input data for named entities, hurtful lexicon, punctuation counts, bigram tf-idf, and empathy ratings
	'''
	# get NER vector 
	# TODO: check with eli whether ner will still work after lemmatization is applied
	nerv = get_ner_matrix(sentences)

	# preprocess to remove quotation marks and lemmatize
	preprocessed_sentences = utils.lemmatize(sentences)

	# create lexical vector
	lv = create_lexical_matrix(preprocessed_sentences, [c for c in punctuation])

	# get empathy vectors
	em = get_empath_ratings(preprocessed_sentences)

	# normalize tf-idf space so that dev and train vector have the same featurize dimensions
	# get tfidf
	tf = get_tfidf(preprocessed_sentences, vectorizer)

	# get hurtlex feature vector
	hv = extract_hurtlex(sentences, hurtlex_dict, hurtlex_cat)

	# normalize the vectors
	print("normalizing vectors...")
	nv = utils.normalize_vector(nerv, lv, em, hv, tf)
	
	return nv
