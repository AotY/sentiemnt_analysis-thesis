#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tokenizer
"""
#  import re
import jieba


class Tokenizer:
    def __init__(self, userdict_path=None):
        # load user dict
        if userdict_path is not None and userdict_path != '':
            jieba.load_userdict(userdict_path)

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        tokens = self.clean_str(text)
        #  tokens = [token for token in tokens if len(token.split()) > 0]
        tokens = [token.split()[0]
                  for token in tokens if len(token.split()) > 0]
        return tokens

    def clean_str(self, text):
        text = text.lower()

        text = text.replace(':', ' : ')
        text = text.replace(',', ' , ')
        text = text.replace('：', ' ： ')
        text = text.replace('，', ' ， ')
        text = text.replace('。', ' 。 ')
        text = text.replace('"', ' " ')
        text = text.replace('“', ' “ ')
        text = text.replace('”', ' ” ')


        tokens = list(jieba.cut(text))
        #  tokens = [token.split()[0] for token in tokens if len(token.split()) > 0]

        if len(tokens) == 0:
            return []

        new_tokens = list()
        for token in tokens:
            try:
                float(token)
                new_tokens.append('<NUMBER>')
            except ValueError as e:
                new_tokens.append(token)
                continue

        return new_tokens
