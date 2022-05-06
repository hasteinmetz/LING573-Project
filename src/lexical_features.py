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


def get_vocabulary(training_sents: List[str]) -> TfidfVectorizer:
    '''Helper function that gets counts of the training data set''' 
    vectorizer = TfidfVectorizer(decode_error='ignore') # ALL WORDS ARE LOWERCASED!
    fitted_vectorizer = vectorizer.fit(training_sents) # DO NOT fit the vectorizer to the test data!
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

    print("create lexical vector...")

    # create lexical vector
    lv = create_lexical_matrix(train_sentences, [c for c in punctuation])
    lv_test = create_lexical_matrix(dev_sentences, [c for c in punctuation])
 
    print("getting tf-idf...")

    # get vocabulary counts (fit the vectorizer)
    vectorizer = get_vocabulary(train_sentences)

    # get tfidf
    tf = get_tfidf(train_sentences, vectorizer)
    tf_test = get_tfidf(dev_sentences, vectorizer)

    print("normalizing vectors...")

    # normalize the vectors
    nv = normalize_vector(lv, tf)
    nv_test = normalize_vector(lv_test, tf_test)
    print(nv)

    # see what svm says
    # vectors, labels = shuffle(nv, train_labels)
    svm = SVC()
    svm.fit(nv, labels)
    print(f'accuracy: {svm.score(nv_test, dev_labels)}')