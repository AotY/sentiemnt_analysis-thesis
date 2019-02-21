#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
preprocessing
"""

import os
import re
import argparse
import random

from tqdm import tqdm
from collections import Counter

from tokenizer import Tokenizer
from utils import save_distribution

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--cleaned_path', type=str, default='')
parser.add_argument('--vocab_freq_path', type=str, default='')
parser.add_argument('--save_dir', type=str, default='./data')

parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--max_len', type=int, default=150)
args = parser.parse_args()

tokenizer = Tokenizer()


def cleaning_stats():
    len_dict = {}
    freq_dict = Counter()

    cleaned_file = open(args.cleaned_path, 'w', encoding='utf-8')
    cleaned_datas = list()
    for dir_name, subdir_list, file_list in os.walk(args.data_dir):
        for file_name in file_list:
            #  print('dir_name ---------> %s' % dir_name)
            #  print('file_name ---------> %s' % file_name)
            #  file_path = os.path.join(os.path.join(args.data_dir, dir_name), file_name)
            file_path = os.path.join(dir_name, file_name)
            print('file_path---------> %s' % file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_set = set()
                for line in tqdm(f):
                    line = line.rstrip()
                    try:
                        disease, doctor, date, score, text = line.split('\t')
                    except ValueError as e:
                        print(line)
                        print(e)
                        continue

                    if not bool(score) or not bool(text):
                        continue

                    unique_line = "{} {} {} {} {}".format(disease, doctor, date, score, text)
                    if unique_line in line_set:
                        continue
                    line_set.add(unique_line)

                    # format date
                    date = re.findall(r'\d+', date)
                    if len(date) > 0:
                        date = int(date[0])

                    # label
                    if score in ['1', '2']:
                        label = 0
                    elif score in ['3']:
                        label = 1
                    elif score in ['4', '5']:
                        label = 2
                    else:
                        raise ValueError('score: %s is not valid.' % score)

                    tokens = tokenizer.tokenize(text)
                    freq_dict.update(tokens)
                    len_dict[len(tokens)] = len_dict.get(len(tokens), 0) + 1

                    if len(tokens) < args.min_len or len(tokens) > args.max_len:
                        continue

                    text = ' '.join(tokens)

                    cleaned_datas.append((disease, doctor, date, label, text))
                del line_set

    # shuffle
    random.shuffle(cleaned_datas)

    # re write
    for item in cleaned_datas:
        disease, doctor, date, label, text = item
        cleaned_file.write('%s\t%s\t%s\t%s\t%s\n' % (
            disease, doctor, date, label, text))

    cleaned_file.close()

    return freq_dict, len_dict


def main():
    freq_dict, len_dict = cleaning_stats()
    freq_list = sorted(freq_dict.items(),
                       key=lambda item: item[1], reverse=True)
    save_distribution(freq_list, args.vocab_freq_path)

    len_list = sorted(len_dict.items(),
                      key=lambda item: item[0], reverse=False)
    save_distribution(len_list, os.path.join(args.save_dir, 'len.dist'))


if __name__ == '__main__':
    main()
