#!/usr/bin/python

import pandas as pd
import numpy as np
import spacy
import utils

def read_from_tsv(lex_data):
    """
    read in and codify hurtlex in the format of 'lemma:[category, tag]'
    """
    output_dict = {}

    df = pd.read_csv(lex_data, sep='\t', header=0, usecols=[1,2,4])
    feature_list = set(df['category'].tolist())

    df['category'] = df[['category', 'pos']].values.tolist()
    output_dict = dict(zip(df.lemma, df.category))

    return output_dict, feature_list


def check_phrase(sentence, lex_dict):
    """
    check phrases in hurtlex
    """
    contain_phrase = False
    tokens = []

    for k, v in lex_dict.items():
        if k in sentence and len(k.split()) > 1:
            contain_phrase = True
            tokens.append(v[0])

    return contain_phrase, tokens


def count_feature(sentence, lex_dict, feature_list, tagger):
    """
    input a sentence, output the encoding of the hurtlex feature
    """
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

#main function
def extract_hurtlex(data_file):
    """
    input data, output a (# of datapoints, 17) ndarray
    """
    #PARAMETER
    input_lex = "/home2/pbban/LING573-Project/src/data/hurtlex_en.tsv"
    #data_file = "/home2/pbban/LING573-Project/src/data/hahackathon_prepo1_dev.csv"

    #set up
    j = 0
    tagger = spacy.load("en_core_web_sm")

    #read in Hurtlex
    lex_dict, feature = read_from_tsv(input_lex)

    #read in data
    sentences, labels = utils.read_data_from_file(data_file)

    features = []
    for data in sentences:
        s = count_feature(data, lex_dict, feature, tagger)
        features.append(s)

    return np.array(features)
