#!/usr/bin/python

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.utils import shuffle
from typing import *
from string import punctuation
from functools import reduce

def create_lexical_matrix(sentences: List[str], chars: list[str]) -> np.ndarray:
    '''Take a dataset and return an [num_sentences, num_chars] matrix of counts
    of the number of times each character appears in a list'''
    # initialize a list to store vectors
    return_vectors = []
    for s in sentences:
        char_counts = {c : s.count(c) for c in chars}
        return_vectors.append(char_counts)
    df = pd.DataFrame(return_vectors)
    return df.to_numpy()


def lemmatize(sentences: List[str]) -> List[str]:
    '''Process the sentence into a string of lemmas (to potentially improve the TF-IDF measure)
    This function requires spacy to use'''
    import spacy
    processer = spacy.load("en_core_web_sm")
    lemmatizer = processer.get_pipe("lemmatizer")
    to_str = lambda x, y: x + " " + y
    lemmatize = lambda x: reduce(to_str, [tkn.lemma_ for tkn in processer(x)])
    new_sents = [lemmatize(x) for x in sentences]
    return new_sents
    

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
    if concat_labels:
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


def normalize_vector(*arrays: np.ndarray) -> np.ndarray:
    '''Take several arrays and concatenate them column-wise before normalizing each row'''
    concatenated = np.concatenate(arrays, axis=1)
    norm = Normalizer()
    return norm.fit(concatenated).transform(concatenated)


if __name__ == '__main__':
    import utils
    import sys

    print("reading sentences...")

    # read in the training and development data
    train_sentences_raw, train_labels = utils.read_data_from_file(sys.argv[1])
    dev_sentences_raw, dev_labels = utils.read_data_from_file(sys.argv[2])

    # preprocess to remove quotation marks
    train_sentences = utils.preprocess_quotes(train_sentences_raw)
    dev_sentences = utils.preprocess_quotes(dev_sentences_raw)

    # preprocess to get just lemmas
    train_sentences = lemmatize(train_sentences)
    dev_sentences = lemmatize(dev_sentences)

    print("create lexical vector...")

    # create lexical vector
    lv = create_lexical_matrix(train_sentences, [c for c in punctuation])
    lv_test = create_lexical_matrix(dev_sentences, [c for c in punctuation])
 
    print("getting tf-idf...")

    # get vocabulary counts (fit the vectorizer)
    vectorizer = get_vocabulary(train_sentences, 'english', concat_labels = train_labels)

    # get tfidf
    tf = get_tfidf(train_sentences, vectorizer)
    tf_test = get_tfidf(dev_sentences, vectorizer)

    print("normalizing vectors...")

    # normalize the vectors
    nv = normalize_vector(lv, tf)
    nv_test = normalize_vector(lv_test, tf_test)
    print(nv)

    # see what svm says
    vectors, labels = shuffle(nv, train_labels)
    svm = SVC()
    svm.fit(nv, train_labels)
    print(f'accuracy: {svm.score(nv_test, dev_labels)}')