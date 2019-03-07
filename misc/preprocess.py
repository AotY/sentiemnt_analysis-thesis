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
parser.add_argument('--userdict', type=str, default='')
parser.add_argument('--label_cleaned_path', type=str, default='')
parser.add_argument('--score_cleaned_path', type=str, default='')
parser.add_argument('--vocab_freq_path', type=str, default='')
parser.add_argument('--save_dir', type=str, default='./data')

parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--max_len', type=int, default=150)
args = parser.parse_args()

tokenizer = Tokenizer(args.userdict)

def cleaning_stats():
    print('cleaning data...')
    len_dict = {}
    freq_dict = Counter()

    label_dict = Counter()

    score_dict = Counter()

    score_files = [None] * 5
    for i in range(5):
        score_files[i] = open(os.path.join(args.save_dir, 'score.%d.txt' % (i+1)), 'w', encoding='utf-8')

    label_files = [None] * 3
    for i in range(3):
        label_files[i] = open(os.path.join(args.save_dir, 'label.%d.txt' % (i+1)), 'w', encoding='utf-8')

    label_cleaned_file = open(args.label_cleaned_path, 'w', encoding='utf-8')
    score_cleaned_file = open(args.score_cleaned_path, 'w', encoding='utf-8')
    cleaned_datas = list()
    error_lines = 0
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
                        #  print(e)
                        error_lines += 1
                        continue

                    if not bool(score) or not bool(text):
                        continue

                    if len(text) < args.min_len or len(text) > args.max_len:
                        continue

                    line_str = "{}_{}_{}_{}".format(disease, doctor, score, text)
                    if line_str in line_set:
                        continue
                    line_set.add(line_str)

                    # format date
                    date = re.findall(r'\d+', date)
                    if len(date) > 0:
                        date = int(date[0])

                    # label
                    if score in ['1', '2']:
                        label = 1
                    elif score in ['3']:
                        label = 2
                    elif score in ['4', '5']:
                        label = 3
                    else:
                        raise ValueError('score: %s is not valid.' % score)

                    label_dict.update(str(label))

                    score_dict.update(str(score))

                    tokens = tokenizer.tokenize(text)
                    freq_dict.update(tokens)

                    len_dict[len(tokens)] = len_dict.get(len(tokens), 0) + 1

                    text = ' '.join(tokens)

                    # split by score
                    # score_files[int(score) - 1].write('%s\t%s\t%s\t%s\t%s\n' % (
                        # disease, doctor, date, score, text.replace(' ', '')))
                    score_files[int(score) - 1].write('%s\t%s\n' % (score, text.replace(' ', '')))

                    label_files[int(label) - 1].write('%s\t%s\n' % (label, text.replace(' ', '')))

                    cleaned_datas.append((disease, doctor, date, score, label, text))
                del line_set

    # shuffle
    random.shuffle(cleaned_datas)
    print('error_lines: %d' % error_lines)

    # re write
    for item in cleaned_datas:
        disease, doctor, date, score, label, text = item
        # label_cleaned_file.write('%s\t%s\t%s\t%s\t%s\n' % (
            # disease, doctor, date, label, text))
        label_cleaned_file.write('%s\t%s\n' % (label, text))

        # score_cleaned_file.write('%s\t%s\t%s\t%s\t%s\n' % (
            # disease, doctor, date, score, text))
        score_cleaned_file.write('%s\t%s\n' % (score, text))

    label_cleaned_file.close()
    score_cleaned_file.close()

    for score_file in score_files:
        score_file.close()

    return freq_dict, len_dict, label_dict, score_dict


def main():
    freq_dict, len_dict, label_dict, score_dict = cleaning_stats()
    freq_list = sorted(freq_dict.items(),
                       key=lambda item: item[1], reverse=True)
    save_distribution(freq_list, args.vocab_freq_path)

    len_list = sorted(len_dict.items(),
                      key=lambda item: item[0], reverse=False)
    save_distribution(len_list, os.path.join(args.save_dir, 'len.dist'), percentage=False)

    label_list = sorted(label_dict.items(),
                      key=lambda item: item[0], reverse=False)
    save_distribution(label_list, os.path.join(args.save_dir, 'label.dist'), percentage=True)

    score_list = sorted(score_dict.items(),
                      key=lambda item: item[0], reverse=False)
    save_distribution(score_list, os.path.join(args.save_dir, 'score.dist'), percentage=True)


if __name__ == '__main__':
    main()
