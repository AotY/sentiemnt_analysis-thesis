#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tokenizer
"""
import re
#  import jieba
import pkuseg


class Tokenizer:
    def __init__(self, userdict_path=None, max_len=150):
        # load user dict
        #  if userdict_path is not None and userdict_path != '':
            #  jieba.load_userdict(userdict_path)
        self.seg = pkuseg.pkuseg()
        self.max_len = max_len
        url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+' # URLs
        self.url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        tokens = self.clean_str(text)
        tokens = [token.split()[0]
                  for token in tokens if len(token.split()) > 0]
        tokens = [token for token in tokens if len(token) <= self.max_len]
        return tokens

    def clean_str(self, text):
        text = text.lower()
        text = re.sub('^', ' ', text)
        text = re.sub('$', ' ', text)

        text = self.url_re.sub(' URL ', text)

        #  tokens = list(jieba.cut(text))
        tokens = self.seg.cut(text)

        text = ' '.join(tokens)
        text = text.replace('URL', '<URL>')
        tokens = text.split()

        if len(tokens) == 0:
            return []

        """
        new_tokens = list()
        for token in tokens:
            try:
                float(token)
                new_tokens.append('<NUMBER>')
            except ValueError as e:
                new_tokens.append(token)
                continue

        return new_tokens
        """
        return tokens
