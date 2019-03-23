#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Convert words to pinyins
"""


#  import os
import argparse
from tqdm import tqdm
#  import snownlp
#  from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--document_split', type=str, default='DOCUMENT_SPLIT')
parser.add_argument('--save_path', type=str, default='')
args = parser.parse_args()


save_file = open(args.save_path, 'w', encoding='utf-8')
none_pinyin = 'none'

with open(args.data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.rstrip()
        if line == args.document_split:
            save_file.write('%s\n' % args.document_split)

        words = line.split()
        pinyins = list()
        for word in words:
            #  pinyin_list = snownlp.normal.get_pinyin(word)
            pinyin_list2 = pypinyin.pinyin(word, style=pypinyin.Style.NORMAL, heteronym=False)
            pinyin_list = [heteronyms[0] for heteronyms in pinyin_list2]
            pinyin_str = ''.join(pinyin_list)
            if pinyin_str != word:
                pinyins.append(pinyin_str)
            else:
                if len(pinyin_str) == 1: # probably it is a Punctuation
                    pinyins.append(pinyin_str)
                else:
                    pinyins.append(none_pinyin)
        save_file.write('%s\n' % ' '.join(pinyins))

        del pinyins

save_file.close()
