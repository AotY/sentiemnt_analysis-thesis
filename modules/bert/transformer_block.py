#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/transformer.py
"""

import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, num_heads, feed_forward_hidden, dropout):
        """
        :param d_model: d_model size of transformer
        :param num_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=num_heads, d_model=d_model)
        # self.attention = MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=feed_forward_hidden, dropout=dropout)

        self.input_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_model, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, max_len, d_model]
            mask: [batch_size, 1, max_len, max_len]
        Returns:
            outputs: [batch_size, max_len, d_model]
            attns: [batch_size, num_heads, max_len, max_len] list
        """
        x, attns = self.attention(x, x, x, mask=mask)
        x = self.input_sublayer(x)
        #  x, attns = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))

        x = self.output_sublayer(x, self.feed_forward)
        x = self.dropout(x)
        return x, attns

