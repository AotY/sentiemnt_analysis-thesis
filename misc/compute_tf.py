#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Compute tf from a single file.
"""

import argparse
import math
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--document_split', type=str, default='DOCUMENTSPLIT')
args = parser.parse_args()

word_counter = Counter()
words = list()
line_count = 0
doc_count = 0

save_file = open(args.save_path, 'w', encoding='utf-8')

with open(args.data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.rstrip()
        if line == args.document_split:
            if line_count > 0:
                # save tf for each document
                doc_count += 1
                tf = sorted(word_counter.items(), key=lambda item: item[1], reverse=False)
                total_count = sum(tf.values())
                for word, count in tf:
                    save_file.write('%s %f\n' % (word, count / total_count))

            word_counter.clear()
            line_count = 0
            save_file.write('%s\n' % args.document_split)
        else:
            line_words = line.split()
            if len(line_words) > 0:
                word_counter.update(line_words)
                line_count += 1


save_file.write('%s\n' % args.document_split)
save_file.close()

print('doc_count: ', doc_count)
print('Compute Success.')
