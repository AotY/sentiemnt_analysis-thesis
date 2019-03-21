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
args = parser.parse_args()

word_counter = Counter()
doc_count = 0
with open(args.data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.rstrip()
        words = set(line.split())
        word_counter.update(words)

        doc_count += 1



# compute idf
idf = {}
for word, count in word_counter.items():
    idf[word] = math.log(doc_count / count)
    #  print('word: %s, idf: %.3f' % (word, idf[word]))

idf = sorted(idf.items(), key=lambda item: item[1], reverse=False)

with open(args.save_path, 'w', encoding='utf-8') as f:
    for word, value in idf:
        f.write('%s\t%.5f\n' % (word, value))

#  pickle.dump(idf, open(args.save_path, 'wb'))

print('Compute Success.')
