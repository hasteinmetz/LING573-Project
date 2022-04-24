#!/usr/bin/env python

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from torch import load as load_binary_file
import argparse
import csv
import numpy as np

def load_embeddings(file: str):
    '''Load pretrained embeddings from a file.'''
    embeddings = load_binary_file(file)
    return np.asarray(embeddings, dtype=np.float32)


def argparser():
    '''Parse the input arguments'''
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--raw_data', help='path to input raw data file')
    parser.add_argument('--input_embeddings', help="path to sentence embeddings file")
    parser.add_argument('--kernel', default='rbf', help="Kernel function used by SVM")
    args = parser.parse_args()
    return args


def load_raw_data(filepath):
    # load sentences and labels from csv
    sentences, labels = [], []
    with open(filepath, 'r') as datafile:
        data = csv.reader(datafile)
        for row in data:
            sentences.append(row[0])
            labels.append(int(row[1]))
    return sentences, np.asarray(labels, dtype=int)

def fit_svm(data, labels, kernel):
    '''Fit the svm model'''
    svm = SVC(kernel=kernel)
    svm.fit(data, labels)
    return svm

def predict(svm, x, labels):
    '''predict the output from train/dev/test samples'''
    
    # predictions
    y_pred = svm.predict(x)

    #accuracy
    accuracy = accuracy_score(labels, y_pred)
    #f1
    f1 = f1_score(labels, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")

if __name__ == '__main__':
    # get input arguments
    args = argparser()

    # load raw data
    sentences, labels = load_raw_data(args.raw_data)

    # load embeddings
    embeddings = load_embeddings(args.input_embeddings)

    # fit svm
    if args.kernel:
        kernel = args.kernel
    else:
        kernel = 'rbf'
    svm = fit_svm(embeddings, labels, kernel)

    # evaluate model on training data
    predict(svm, embeddings, labels)

   