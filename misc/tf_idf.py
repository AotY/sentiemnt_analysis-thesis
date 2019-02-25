#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
TF-IDF
https://github.com/Jasonnor/tf-idf-python/blob/master/src/tf_idf.py
"""

import jieba
import os
from math import log
from operator import itemgetter


class TFIDF:
    def __init__(self, stop_words_path=None):
        """
        stop_words_path: the path of stop_words
        """
        self.files = {}
        self.corpus = {}
        self.stop_words = set()
        if stop_words_path is not None:
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    self.stop_words.add(line)

    def add_file(self, file_names):
        for file_name in file_names:
            self.add_file(file_name)

    def add_file(self, file_name):
        # Load data and cut
        content = open(file_name, 'r', encodeing='utf-8').read()
        words = list(jieba.cut(content))

        # Build dictionary
        dictionary = {}
        for word in words:
            if len(word.strip()) < 1 or word.lower() in self.stop_words:
                continue
            dictionary[word] = dictionary.get(word, 0.0) + 1.0
            self.corpus[word] = self.corpus.get(word, 0.0) + 1.0

        # Get term frequency
        total = sum(dictionary.values())
        for k in dictionary:
            dictionary[k] /= total

        # Add tf to the corpus
        self.files[file_name] = dictionary

    def get_tf_idf(self, file_name, top_k):
        # Get inverse document frequency
        tf_idf_of_file = {}
        for w in self.corpus.keys():
            w_in_f = 1.0
            for f in self.files:
                if w in self.files[f]:
                    w_in_f += 1.0
            # Get tf-idf
            if w in self.files[file_name]:
                tf_idf_of_file[w] = log(len(self.files) / w_in_f) * self.files[file_name][w]
        # Top-K result of tf-idf
        tags = sorted(tf_idf_of_file.items(), key=itemgetter(1), reverse=True)
        return tags[:top_k]

    def similarities(self, list_of_words):
        # Building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # Normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # Get the list of similarities
        sims = []
        for f in self.files:
            score = 0.0
            for k in query_dict:
                if k in self.files[f]:
                    score += (query_dict[k] / self.corpus[k]) + (self.files[f][k] / self.corpus[k])
            sims.append([f, score])

        return sorted(sims, key=itemgetter(1), reverse=True)


if __name__ == "__main__":
    table = TFIDF()
    folder_name = '笑傲江湖'
    dir = os.path.dirname(__file__)
    folder = os.path.join(dir, '../data/' + folder_name)
    num_of_files = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]) + 1

    for x in range(1, num_of_files):
        file_name = folder_name + '/' + str(x).zfill(2) + '.txt'
        table.add_file(file_name)

    top_k = 20
    for x in range(1, num_of_files):
        target_file = folder_name + '/' + str(x).zfill(2) + '.txt'
        print('Top ' + str(top_k) + ' of tf-idf in ' + target_file + ' : ')
        print(table.get_tf_idf(target_file, top_k))
        print()

    key_word = '任我行'
    print('tf-idf of key word "' + key_word + ' : ')

