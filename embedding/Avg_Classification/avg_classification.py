#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Embedding Avg as features

"""


import os
import sys
import time
import random
import argparse
from tqdm import tqdm
from gensim.models import KeyedVectors

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='')
parser.add_argument('--embedding_path', type=str, help='')
parser.add_argument('--test_size', type=float, default=0.2, help='')
parser.add_argument('--min_len', type=int, default=3, help='')
args = parser.parse_args()

print(' '.join(sys.argv))

embedding = KeyedVectors.load_word2vec_format(args.embedding_path, binary=True)

def tokens_avg_vector(tokens):
    vectors = list()
    for token in tokens:
        if token in embedding.vocab:
            vector = embedding[token]
            vectors.append(vector)
    if len(vectors) == 0:
        return None

    vectors = np.array(vectors, dtype=np.float32)
    avg_vector = vectors.mean(axis=0)
    return avg_vector


def read_data():
    X = list()
    y = list()
    with open(args.data_path, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            label, text = line.split('\t')
            tokens = text.split()
            tokens = [token.split()[0]
                      for token in tokens if len(token.split()) > 0]
            if len(tokens) < args.min_len:
                continue

            avg_vector = tokens_avg_vector(tokens)
            if avg_vector is None:
                continue
            X.append(avg_vector)
            y.append(int(label))

    X = np.array(X)
    y = np.array(y)
    return X, y


X, y = read_data()
lr = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=23)

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

print(classification_report(y_test, y_predict))
