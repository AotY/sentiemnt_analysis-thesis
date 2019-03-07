#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/bert.py
"""

import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from ..utils import LayerNorm


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embedding_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embedding_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embedding_size=embedding_size)
        self.position = PositionalEmbedding(d_model=embedding_size)
        self.segment = SegmentEmbedding(embedding_size=embedding_size)

        self.norm = LayerNorm(embedding_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embedding_size = embedding_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        x = self.norm(x)
        return self.dropout(x)


