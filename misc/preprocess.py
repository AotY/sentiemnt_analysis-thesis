#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
preprocessing
"""

import argparse

import pandas as pd
from tqdm import tqdm

from tokenizer import Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--raw_data', type=str, default='')
parser.add_argument('--text', type=str, default='review')
parser.add_argument('--label', type=str, default='label')
parser.add_argument('--user_dict', type=str, default='')
parser.add_argument('--max_string_len', type=int, default=30)
parser.add_argument('--tokenizer_name', type=str, default='thulac', help='which tokenizer')
parser.add_argument('--save_path', type=str, default='')

parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--max_len', type=int, default=250)
args = parser.parse_args()

tokenizer = Tokenizer(args.tokenizer_name, args.user_dict, args.max_string_len)


def main():
    save_file = open(args.save_path, 'w', encoding='utf-8')
    df = pd.read_csv(args.raw_data)
    df = df.dropna(how='any')

    for text, label in tqdm(zip(df[args.text], df[args.label])):
        if len(text) < args.min_len and len(text) > args.max_len:
            continue

        label = int(label)
        if args.label == 'rating':
            if label == 5:
                label = 2 # positive
            elif label == 3:
                label = 1 # neutral
            elif label == 1:
                label = 0 # negative
            else:
                continue

        words = tokenizer.tokenize(text)
        if len(words) == 0:
            continue
        save_file.write('%d\t%s\n' % (label, ' '.join(words)))

    save_file.close()


if __name__ == '__main__':
    main()
