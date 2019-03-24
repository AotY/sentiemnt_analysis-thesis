#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Stats
"""

import os
#  import re
import argparse
#  import random

from tqdm import tqdm
from collections import Counter

#  from tokenizer import Tokenizer
from utils import save_distribution

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_dir', type=str, default='./data')

args = parser.parse_args()
data_name = args.data_path.split('/')[-1].split('.')[0]
print('data_name: %s' % data_name)


def stats():
    len_dict = {}
    freq_dict = Counter()

    label_dict = Counter()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()

            label, words = line.split('\t')
            words = words.split()

            label_dict.update(str(label))

            freq_dict.update(words)

            len_dict[len(words)] = len_dict.get(len(words), 0) + 1

    return freq_dict, len_dict, label_dict


def main():
    freq_dict, len_dict, label_dict = stats()
    freq_list = sorted(freq_dict.items(),
                       key=lambda item: item[1], reverse=True)
    save_distribution(freq_list, os.path.join(
        args.save_dir, '%s.vocab_freq.dist' % data_name))

    len_list = sorted(len_dict.items(),
                      key=lambda item: item[0], reverse=False)
    save_distribution(len_list, os.path.join(
        args.save_dir, '%s.len.dist' % data_name), percentage=False)

    label_list = sorted(label_dict.items(),
                        key=lambda item: item[0], reverse=False)
    save_distribution(label_list, os.path.join(
        args.save_dir, '%s.label.dist' % data_name), percentage=True)


if __name__ == '__main__':
    main()
