#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Build vocab Pretrained word embedding
"""

import argparse
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
from os.path import exists

from vocab import Vocab

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_path', type=str, help='')
parser.add_argument('--wordvec_file', type=str, help='')
parser.add_argument('--type', type=str, help='')
parser.add_argument('--embedding_size', type=int, help='')
parser.add_argument('--save_path', type=str, help='')

args = parser.parse_args()

vocab = Vocab()
vocab.load(args.vocab_path)
args.vocab_size = int(vocab.size)
print('vocab size: ', args.vocab_size)


def load_word2vec():
    if not exists(args.wordvec_file):
        raise Exception(
            "You must download word vectors through `download_wordvec.py` first")
    word2vec = KeyedVectors.load_word2vec_format(
        args.wordvec_file, binary=True)

    word_vectors = []
    miss_count = 0
    # for id, word in tqdm(vocab.idx2word.items()):
    items = sorted(vocab.idx2word.items(), key=lambda x: x[0])
    for id, word in items:
        # print(id, word)
        if word in word2vec.vocab:
            vector = word2vec[word]
        else:
            vector = np.random.normal(
                scale=0.20, size=args.embedding_size)  # random vector
            miss_count += 1

        word_vectors.append(vector)
    weight = np.stack(word_vectors)
    np.save(args.save_path, weight)
    print('miss_count:', miss_count)
    #  return weight


def load_glove():
    if not exists(args.wordvec_file):
        raise Exception(
            "You must download word vectors through `download_wordvec.py` first")

    glove_model = {}
    with open(args.wordvec_file) as file:
        for line in file:
            line_split = line.split()
            word = ' '.join(line_split[:-args.embedding_size])
            numbers = line_split[-args.embedding_size:]
            glove_model[word] = numbers
    glove_vocab = glove_model.keys()

    word_vectors = []
    for word in args.vocab_words:

        if word in glove_vocab:
            vector = np.array(glove_model[word], dtype=float)
        else:
            vector = np.random.normal(
                scale=0.2, size=args.embedding_size)  # random vector

        word_vectors.append(vector)

    weight = np.stack(word_vectors)
    np.save(args.save_path, weight)
    #  return weight


if __name__ == "__main__":
    if args.type == 'glove':
        load_glove()
    else:
        load_word2vec()
