#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Compute idf from a single file.
see each line is a document.
"""

import argparse
import math
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--max_string_len', type=int, default=30)
parser.add_argument('--min_count', type=int, default=2)
parser.add_argument('--document_split', type=str, default='DOCUMENTSPLIT')
args = parser.parse_args()

word_counter = Counter()
words = list()
doc_count = 0
line_count = 0
with open(args.data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.rstrip()
        if line == args.document_split:
            if line_count > 0:
                doc_count += 1
                word_counter.update(set(words))
            words.clear()
            line_count = 0
        else:
            line_words = line.split()
            if len(line_words) > 0:
                words.extend(line_words)
                line_count += 1

# compute idf
idf = {}
for word, count in word_counter.items():
    if len(word) >= args.max_string_len:
        continue
    if count <= args.min_count:
        continue
    idf[word] = math.log(doc_count / count)
    #  print('word: %s, idf: %.3f' % (word, idf[word]))

idf = sorted(idf.items(), key=lambda item: item[1], reverse=False)

idf_len = len(idf)
with open(args.save_path, 'w', encoding='utf-8') as f:
    f.write('%d\n' % idf_len)
    for word, value in idf:
        f.write('%s %f\n' % (word, value))

#  pickle.dump(idf, open(args.save_path, 'wb'))
print('Compute Success.')
