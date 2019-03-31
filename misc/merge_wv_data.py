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
import json
import argparse
import pandas as pd
#  import numpy as np
#  from tqdm import tqdm
from tokenizer import Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--document_split', type=str, default='DOCUMENTSPLIT')
parser.add_argument('--min_len', type=int, default=5)
#  parser.add_argument('--max_len', type=int, default=2000)
parser.add_argument('--max_string_len', type=int, default=30)
parser.add_argument('--user_dict', type=str, default='')
parser.add_argument('--tokenizer_name', type=str, default='thulac', help='which tokenizer')

args = parser.parse_args()


tokenizer = Tokenizer(args.tokenizer_name, args.user_dict, args.max_string_len)

data_sources = [
    #  'ChnSentiCorp_htl_all.csv',  # 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
    #  'waimai_10k.csv',  # 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条
    #  'online_shopping_10_cats.csv',  # 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，
    #  'zhidao_filter.1.csv',  # 百度知道数据
    #  'zhidao_filter.csv',  # 百度知道数据
    #  'simplifyweibo_4_moods.csv',  # 0 万多条，带情感标注 新浪微博，正负向评论约各 5 万条
    #  'dmsc_v2/ratings.csv',  # 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据
    #  'yf_dianping/ratings.csv',  # 24 万家餐馆，54 万用户，440 万条评论/评分数据
    #  'yf_amazon/ratings.csv',  # 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据
    #  'weibo_senti_100k.csv',  # 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条
    #  'baike_qa2019/baike_qa_train.json',  #
    #  'webtext2019zh/web_text_zh_train.json',
    'wiki_zh',
    'news2016zh/news2016zh_train.json',
    #  'label.cleaned.txt',
    #  'raw.question.txt',
]

save_file = open(args.save_path, 'w', encoding='utf-8')
for source in data_sources:
    print('source: %s' % source)
    #  if not source.startswith('./'):
    #  data_path = os.path.join(args.data_dir, source)
    #  else:
    #  data_path = source
    source_path = os.path.join(args.data_dir, source)
    if source.endswith('csv'):
        df = pd.read_csv(source_path)
        if source.startswith('zhidao_filter'):
            #  df = df.dropna(how='any')
            pass
        else:
            df = df.dropna(how='any')

    if source.startswith('dmsc_v2') or source.startswith('yf_dianping') or source.startswith('yf_amazon'):
        texts = list()
        for comment in df['comment']:
            if len(comment) > args.min_len:
                #  texts.append(comment)
                #  texts.append(args.document_split)
                words = tokenizer.tokenize(comment)
                save_file.write('%s\n' % ' '.join(words))
                #  save_file.write('%s\n' % comment)
                save_file.write('%s\n' % args.document_split)

    elif source.startswith('zhidao_filter'):
        texts = list()
        for items in df.values:
            if items[-1] == 0 or pd.isna(items[2]):
                continue
            if not pd.isna(items[0]):
                query = items[0]
            if not pd.isna(items[1]):
                if len(query) < len(items[1]):
                    query = items[1]
            answer = items[2]

            if len(query) > args.min_len and len(answer) > args.min_len:
                #  texts.append(query + "\t" + answer)
                #  texts.append(args.document_split)
                words = tokenizer.tokenize(query + '\t' + answer)
                save_file.write('%s\n' % ' '.join(words))
                #  save_file.write('%s\n' % (query + '\t' + answer))
                save_file.write('%s\n' % args.document_split)

    elif source.endswith('baike_qa_train.json'):
        texts = list()
        with open(source_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = json.loads(line)
                title = line['title']
                desc = line['desc']
                answer = line['answer']
                if len(title) > len(desc):
                    query = title
                else:
                    query = desc
                if len(query) > args.min_len and len(answer) > args.min_len:
                    #  texts.append(query + "\t" + answer)
                    #  texts.append(args.document_split)
                    words = tokenizer.tokenize(query + '\t' + answer)
                    save_file.write('%s\n' % ' '.join(words))
                    #  save_file.write('%s\n' % (query + '\t' + answer))
                    save_file.write('%s\n' % args.document_split)

    elif source.endswith('web_text_zh_train.json'):
        texts = list()
        with open(source_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = json.loads(line)
                title = line['title']
                desc = line['desc']
                content = line['content']
                if len(desc) > args.min_len and len(desc) != len(title):
                    query = title + ' ' + desc
                if len(query) > args.min_len and len(content) > args.min_len:
                    #  texts.append(query + "\t" + content)
                    #  texts.append(args.document_split)
                    words = tokenizer.tokenize(query + '\t' + content)
                    save_file.write('%s\n' % ' '.join(words))
                    #  save_file.write('%s\n' % (query + '\t' + content))
                    save_file.write('%s\n' % args.document_split)

    elif source.endswith('news2016zh_train.json'):
        texts = list()
        with open(source_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = json.loads(line)
                #  content = line['content']
                texts = line['content'].split()
                if len(texts) == 0:
                    continue
                for text in texts:
                    if len(text) > args.min_len:
                        #  texts.append(content)
                        #  texts.append(args.document_split)
                        try:
                            words = tokenizer.tokenize(text)
                            save_file.write('%s\n' % ' '.join(words))
                            #  save_file.write('%s\n' % (content))
                        except Exception as e:
                            print('text: %s' % text)
                            continue
                save_file.write('%s\n' % args.document_split)

    elif source.startswith('wiki_zh'):
        texts = list()
        for folder in os.listdir(source_path):
            folder_path = os.path.join(source_path, folder)
            #  print(folder_path)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    #  print(file_name)
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.rstrip()
                            line = json.loads(line)
                            texts = line['text'].split()
                            if len(texts) == 0:
                                continue
                            for text in texts:
                                if len(text) > args.min_len:
                                    #  texts.append(text)
                                    #  texts.append(args.document_split)
                                    words = tokenizer.tokenize(text)
                                    save_file.write('%s\n' % ' '.join(words))
                                    #  save_file.write('%s\n' % (text))
                            save_file.write('%s\n' % args.document_split)
    elif source.startswith('new2016zh'):
        texts = list()
        with open(source_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                line = json.loads(line)
                #  title = line['title']
                #  desc = line['desc']
                #  content = line['content']
                #  if len(title) > args.min_len:
                #  texts.append(title)
                #  if len(desc) > args.min_len:
                #  texts.append(desc)
                texts = line['content'].split()
                if len(texts) == 0:
                    continue
                for text in texts:
                    if len(text) > args.min_len:
                        #  texts.append(text)
                        #  texts.append(args.document_split)
                        words = tokenizer.tokenize(text)
                        save_file.write('%s\n' % ' '.join(words))
                        #  save_file.write('%s\n' % (text))
                save_file.write('%s\n' % args.document_split)

    elif source.endswith('raw.question.txt'):
        texts = list()
        print('Loading raw.question.txt...')
        with open(source_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                try:
                    q_id, d_id, title, query, response, \
                        sub, gender, age, onset, labels = line.split('SPLIT')
                except ValueError as e:
                    continue
                if len(query) > args.min_len and len(response) > args.min_len:
                    #  texts.append(query + '\t' + response)
                    #  texts.append(args.document_split)
                    words = tokenizer.tokenize(query + '\t' + response)
                    save_file.write('%s\n' % ' '.join(words))
                    #  save_file.write('%s\n' % (query + '\t' + response))
                    save_file.write('%s\n' % args.document_split)

    elif source.endswith('label.cleaned.txt'):
        texts = list()
        print('Loading label.cleaned.txt...')
        with open(source_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                try:
                    disease, doctor, date, score, text = line.split('\t')
                except ValueError as e:
                    continue
                if not bool(score) or not bool(text):
                    continue
                text = text.rstrip()
                if len(text) > args.min_len:
                    #  texts.append(text)
                    #  texts.append(args.document_split)
                    words = tokenizer.tokenize(text)
                    save_file.write('%s\n' % ' '.join(words))
                    #  save_file.write('%s\n' % (text))
                    save_file.write('%s\n' % args.document_split)
    else:
        texts = list()
        for review in df['review']:
            if len(review) > args.min_len:
                #  texts.append(review)
                #  texts.append(args.document_split)
                words = tokenizer.tokenize(review)
                save_file.write('%s\n' % ' '.join(words))
                #  save_file.write('%s\n' % (review))
                save_file.write('%s\n' % args.document_split)

    """
    print('Save text...')
    #  for text in tqdm(texts):
    for text in texts:
        #  print(text)
        if len(text) > args.min_len + 1:
            # tokenizer
            if text == args.document_split:
                save_file.write('%s\n' % text)
            else:
                words = tokenizer.tokenize(text)
                save_file.write('%s\n' % ' '.join(words))
    del texts
    """

save_file.write('%s\n' % args.document_split)
save_file.close()
