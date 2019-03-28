#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tokenizer
"""
import re
import jieba
import pkuseg
import thulac


class Tokenizer:
    def __init__(self, tokenizer_name='thulac', user_dict=None, max_string_len=30):
        self.tokenizer_name = tokenizer_name
        if tokenizer_name == 'thulac':
            if user_dict is not None and user_dict != '':
                self.seg = thulac.thulac(seg_only=True, user_dict=user_dict)
            else:
                self.seg = thulac.thulac(seg_only=True)
        elif tokenizer_name == 'pkuseg':
            if user_dict is not None and user_dict != '':
                self.seg = pkuseg.pkuseg(user_dict=user_dict)
            else:
                self.seg = pkuse.pkuseg()
        elif tokenizer_name == 'jieba':
            self.seg = jieba
            if user_dict is not None and user_dict != '':
                self.seg.load_userdict(user_dict)

        self.max_string_len = max_string_len
        url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+' # URLs
        tag_str = r'<\w+>'
        self.url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)
        self.tag_re = re.compile(tag_str, re.VERBOSE | re.IGNORECASE)

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        tokens = self.clean_str(text)
        return tokens

    def clean_str(self, text):
        text = text.lower()
        text = re.sub('^', ' ', text)
        text = re.sub('$', ' ', text)

        text = self.url_re.sub(' URL ', text)
        text = self.tag_re.sub(' TAG ', text)

        if self.tokenizer_name in ['jieba', 'pkuseg']:
            text = ' '.join(self.seg.cut(text))
        elif self.tokenizer_name == 'thulac':
            text = self.seg.cut(text, text=True)

        text = text.replace('URL', '<URL>')
        text = text.replace('TAG', '<TAG>')

        tokens = text.split()

        tokens = [token.split()[0] for token in tokens if len(token.split()) > 0]
        tokens = [token for token in tokens if len(token) <= self.max_string_len]

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
