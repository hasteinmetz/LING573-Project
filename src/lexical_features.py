#!/usr/bin/env python

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


def get_tfidf(sentences: List[str], labels: List[int]) -> Dict[int, np.ndarray]:
    '''Get a TF-IDF matrix for each label
    Returns a dictionary where each TF-IDF matrix can be access'''
    d = {'sents': sentences, 'labels': labels}
    df = pd.DataFrame(d)
    matrices = [] # to store the matrices of each class's TF-IDF values
    vectorizer = TfidfVectorizer(decode_error='ignore') # ALL WORDS ARE LOWERCASED!
    # fit the vectorizer vocabulary to all sentences, but DON'T transform it with counts yet
    fitted_vectorizer = vectorizer.fit(sentences)
    for l in set(labels):
        sentences_with_label = df[df['labels'] == l]
        matrix = fitted_vectorizer.transform(sentences_with_label['sents'])
        matrices.append(matrix.toarray())
    return np.concatenate(matrices, axis=0)


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

    # preprocess to remove quotation marks
    train_sentences = utils.preprocess_quotes(train_sentences_raw)

    print("create lexical vector...")

    # create lexical vector
    lv = create_lexical_matrix(train_sentences, [c for c in punctuation])
    print(lv)

    print("getting tf-idf...")

    # get tfidf
    tf = get_tfidf(train_sentences, train_labels)

    print("normalizing vectors...")

    # normalize the vectors
    nv = normalize_vector(lv, tf)
    print(nv)

    # see what svm says
    vectors, labels = shuffle(nv, train_labels)
    svm = SVC()
    svm.fit(vectors[400:,:], labels[400:])
    print(f'accuracy: {svm.score(vectors[0:400,:], labels[0:400])}')