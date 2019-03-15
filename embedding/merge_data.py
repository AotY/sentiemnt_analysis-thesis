#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Merge data from different sources:
    https://github.com/SophonPlus/ChineseNlpCorpus
"""


import os
import argparse
import pandas as pd
from tqdm import tqdm
from misc.tokenizer import Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--userdict', type=str, default='./data/userdict.txt')
parser.add_argument('--save_path', type=str, default='')
args = parser.parse_args()


tokenizer = Tokenizer(args.userdict)

data_sources = [
    'ChnSentiCorp_htl_all.csv',  # 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
    'waimai_10k.csv',  # 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条
    'online_shopping_10_cats.csv',  # 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，
    'weibo_senti_100k.csv',  # 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条
    'simplifyweibo_4_moods.csv',  # 0 万多条，带情感标注 新浪微博，正负向评论约各 5 万条
    'dmsc_v2/ratings.csv',  # 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据
    'yf_dianping/ratings.csv',  # 24 万家餐馆，54 万用户，440 万条评论/评分数据
    'yf_amazon/ratings.csv',  # 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据
    './data/raw.question.txt',
    './data/label.cleaned.txt',
    #  'ez_douban/ratings.csv',  # 5 万多部电影（3 万多有电影名称，2 万多没有电影名称），2.8 万 用户，280 万条评分数据
]

save_file = open(args.save_path, 'w', encoding='utf-8')
for source in data_sources:
    if not source.startswith('./'):
        data_path = os.path.join(args.data_dir, source)
    else:
        data_path = source

    if source.endswith('csv'):
        df = pd.read_csv(data_path)
        df = df.dropna(how='any')
    if source.startswith('dmsc_v2'):
        texts = df['comment']
    elif source.startswith('yf_dianping'):
        texts = df['comment']
    elif source.startswith('yf_amazon'):
        texts = df['comment']
    elif source.startswith('raw.question.txt'):
        texts = list()
        print('Loading raw.question.txt...')
        with open(data_path, 'r') as f:
            #  for line in tqdm(f):
            for line in f:
                line = line.rstrip()
                try:
                    q_id, d_id, title, query, response, \
                        sub, gender, age, onset, labels = line.split('SPLIT')
                except ValueError as e:
                    continue
                texts.append(query)
                texts.append(response)
        pass
    elif source.startswith('label.cleaned.txt'):
        texts = list()
        print('Loading label.cleaned.txt...')
        with open(data_path, 'r') as f:
            #  for line in tqdm(f):
            for line in f:
                line = line.rstrip()
                try:
                    disease, doctor, date, score, text = line.split('\t')
                except ValueError as e:
                    continue
                if not bool(score) or not bool(text):
                    continue

                text = text.rstrip().replace(' ', '').replace(' ', '')
                texts.append(text)

    else:
        texts = df['review']
        pass

    print('Save text...')
    #  for text in tqdm(texts):
    for text in texts:
        #  print(text)
        if len(text) > 5:
            # tokenizer
            words = tokenizer.tokenize(text)
            save_file.write('%s\n' % ' '.join(words))
    texts = None

save_file.close()
