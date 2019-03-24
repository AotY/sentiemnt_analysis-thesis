#! /usr/bin/env python
# -*- coding: utf-8 -*- Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import sys
import argparse
from tqdm import tqdm
from .vocab import Vocab

def read_distribution(vocab_freq_path):
    freq_list = []
    with open(vocab_freq_path, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            try:
                word, freq = line.split()
                freq = int(freq)
                freq_list.append((word, freq))
            except Exception as e:
                print(line)
                print(e)

    return freq_list


def build_vocab(vocab_freq_path, vocab_size=30000, min_count=1):
    vocab_size = int(vocab_size)
    freq_list = read_distribution(vocab_freq_path)

    if min_count > 0:
        print('Clip tokens by min_count')
        freq_list = [item for item in freq_list if item[1] > min_count]

    if vocab_size > 0 and len(freq_list) >= vocab_size:
        print('Clip tokens by vocab_size')
        freq_list = freq_list[:vocab_size]

    vocab = Vocab()
    vocab.build_from_freq(freq_list)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_freq_path', type=str, help='')
    parser.add_argument('--vocab_size', type=float, default=6e4)
    parser.add_argument('--min_count', type=int, default=3)
    parser.add_argument('--vocab_path', type=str, default='')

    args = parser.parse_args()


    vocab = build_vocab(args.vocab_freq_path, args.vocab_size, args.min_count)
    print('vocab_size: ', vocab.size)
    vocab.save(args.vocab_path)

