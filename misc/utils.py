#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Utils
"""
import os
import sys

#  import torch
import numpy as np


def save_distribution(dist_list, save_path, percentage=False):
    if percentage:
        total = 0
        for i, j in dist_list:
            total += j

    with open(os.path.join(save_path), 'w', encoding="utf-8") as f:
        for i, j in dist_list:
            if percentage:
                f.write('{}\t{}\t{per: 3.3f}%\n'.format(
                    i, j, per=(j/total)*100))
            else:
                f.write('%s\t%s\n' % (str(i), str(j)))


def load_glove_embeddings(path, word2idx, embedding_dim):
    """Loading the glove embeddings"""
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        #  return torch.from_numpy(embeddings).float()
        return embeddings
