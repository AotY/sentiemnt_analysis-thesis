#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Define the sub layers in encoder (or decoder) layer.
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
"""

import numpy as np
import torch
import torch.nn as nn

from modules.transformer.sdpa import ScaleDotProductAttention
from modules.utils import init_wt_normal

class  MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_heads

        self.k_size = config.k_size
        self.v_size = config.v_size

        self.q_linear = nn.Linear(config.embedding_size, config.num_heads * config.k_size)
        self.k_linear = nn.Linear(config.embedding_size, config.num_heads * config.k_size)
        self.v_linear = nn.Linear(config.embedding_size, config.num_heads * config.v_size)

        init_wt_normal(self.q_linear.weight, (config.embedding_size + config.k_size))
        init_wt_normal(self.k_linear.weight, (config.embedding_size + config.k_size))
        init_wt_normal(self.v_linear.weight, (config.embedding_size + config.v_size))

        self.attention = ScaleDotProductAttention(
            temperature=np.power(config.k_size, 0.5),
            dropout=config.dropout
        )

        # Applies Layer Normalization over a mini-batch of inputs
        self.layer_norm = nn.LayerNorm(config.embedding_size)

        self.fc = nn.Linear(config.num_heads * config.v_size, config.embedding_size)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, max_len, embedding_size]
            k: [batch_size, max_len, embedding_size]
            v: [batch_size, max_len, embedding_size]
            mask: [batch_size, max_len,embedding_size]
        Returns:
            outputs: [batch_size, len_q, embedding_size]
            attns: [num_heads * batch_size, len_q, len_k]

        """
        residual = q

        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        # print('q shape: ', q.shape)
        q = self.q_linear(q).view(batch_size, len_q, self.num_heads, self.k_size) #[16, 35, 8 * 32]
        k = self.k_linear(k).view(batch_size, len_k, self.num_heads, self.k_size)
        v = self.v_linear(v).view(batch_size, len_v, self.num_heads, self.v_size)
        #  print('q size: ', q.shape)
        #  print('k size: ', k.shape)
        #  print('v size: ', v.shape)

        # [num_heads, batch_size, len, k] -> [num_heads * batch_, len, k]
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.k_size) # # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.k_size)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.v_size)
        #  print('q size: ', q.shape) # [1024, 50, 64]
        #  print('k size: ', k.shape)
        #  print('v size: ', v.shape)

        mask = mask.repeat(self.num_heads, 1, 1) # (n*b) x .. x ..

        #  outputs: [num_heads * batch_size, len_q, v_size]
        #  attns: [num_heads * batch_size, len_q, len_k]
        outputs, attns = self.attention(q, k, v, mask=mask)

        outputs = outputs.view(self.num_heads, batch_size, len_q, self.v_size)

        # [batch_size, len_q, num_heads * v_size]
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)

        # [batch_sizes, len_q, embedding_size]
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)

        outputs = self.layer_norm(outputs + residual)

        return outputs, attns


class PositionwiseFeedForward(nn.Module):
    """A two feed forward layer module."""
    def __init__(self,
                config):
        super().__init__()

        self.cnn1 = nn.Conv1d(config.embedding_size, config.inner_hidden_size, 1) # position wise
        self.cnn2 = nn.Conv1d(config.inner_hidden_size, config.embedding_size, 1) # position wise

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, max_len, embedding_size]
        Return: 
            outputs: [batch_size, max_len, embedding_size]
        """
        residual = inputs

        # [batch_size, embedding_size, max_len]
        outputs = inputs.transpose(1, 2)

        # [batch_size, inner_hidden_size, max_len]
        outputs = self.cnn1(outputs)
        outputs = torch.relu(outputs)

        # [batch_size, embedding_size, max_len]
        outputs = self.cnn2(outputs)

        # [batch_size, max_len, embedding_size]
        outputs = outputs.transpose(1, 2)

        outputs = self.dropout(outputs)

        outputs = self.layer_norm(outputs + residual)

        return outputs

