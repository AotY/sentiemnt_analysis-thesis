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

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
args = parser.parse_args()

def main():
    label_sets = [set(), set(), set()]

    # label_files = [None] * 3
    # for i in range(3):
        # label_files[i] = open(os.path.join(args.save_dir, 're.label.%d.txt' % (i+1)), 'w', encoding='utf-8')

    re_cleaned_file = open(os.path.join(args.save_dir, 'label.re.cleaned.txt'), 'w', encoding='utf-8')

    with open(args.data_path, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            # disease, doctor, date, label, text = line.split('\t')
            label, text = line.split('\t')

            label = int(label)
            label_sets[label - 1].add(text)

    for i, label_set in enumerate(label_sets):
        for text in label_set:
            # if i == 0:
                # if text not in label_sets[1] and text not in label_sets[2]:
                    # label_files[i].write('%s\t%s\n' % ((i+1), text))
                    # re_cleaned_file.write('%s\t%s\n' % ((i+1), text))
            # elif i == 1:
                # if text not in label_sets[0] and text not in label_sets[2]:
                    # label_files[i].write('%s\t%s\n' % ((i+1), text))
                    # re_cleaned_file.write('%s\t%s\n' % ((i+1), text))
            if i == 2:
                if text not in label_sets[0] and text not in label_sets[1]:
                    # label_files[i].write('%s\t%s\n' % ((i+1), text))
                    re_cleaned_file.write('%s\t%s\n' % ((i+1), text))
            else:
                re_cleaned_file.write('%s\t%s\n' % ((i+1), text))


    re_cleaned_file.close()
    # for label_file in label_files:
        # label_file.close()


if __name__ == "__main__":
    main()
