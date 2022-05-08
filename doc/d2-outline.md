# D3 Outline

## Abstract

## Introduction

## Task Description

## System Overview
- Late fusion ensemble model of RoBERTa and feature classifier
  - Lexical features included:
    - Preprocessing:
      - Words are lemmatized to reduce computational complexity.
    - Punctuation
    - TF-IDF
      - Preprocessing for TF-IDF only:
        - All sentences for each label are concatenated and then TF-IDF vectorizer is done.
        - Stop words are removed to improve results and reduce computation.
        - Words are lowercased as well.
  - [EMPATH](https://github.com/Ejhfast/empath-client) maybe?
    - Assigns a score for different emotional/cognitive/social categories.
    - Has its own preprocessing/dictionary.

## Approach

## Results

## Discussion

## Conslusion
