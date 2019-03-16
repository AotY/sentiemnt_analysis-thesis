#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Convert words to pinyins
"""


import os
import argparse
from tqdm import tqdm
#  import snownlp
#  from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
args = parser.parse_args()


save_file = open(args.save_path, 'w', encoding='utf-8')

with open(args.data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.rstrip()
        words = line.split()
        pinyins = list()
        pinyin = 'none'
        for word in words:
            #  pinyin_list = snownlp.normal.get_pinyin(word)
            pinyin_list2 = pypinyin.pinyin(word, style=pypinyin.Style.NORMAL, heteronym=False)
            pinyin_list = [heteronyms[0] for heteronyms in pinyin_list2]
            pinyin_str = ''.join(pinyin_list)
            if pinyin_str != word:
                pinyin = pinyin_str
            pinyins.append(pinyin)

        save_file.write('%s\n' % ' '.join(pinyins))



save_file.close()
