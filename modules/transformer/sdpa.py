#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scale Dot-Product Attention
"""

import torch
import torch.nn as nn
import numpy as np


class ScaleDotProductAttention(nn.Module):
    def __init__(self,
                 temperature,
                 dropout=0.1):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [num_heads * batch_size, len_q, k_size]
            k: [num_heads * batch_size, len_k, k_size]
            v: [num_heads * batch_size, len_v, v_size]

        Returns:
            outputs: [num_heads * batch_size, len_q, v_size]
            attns: [num_heads * batch_size, len_q, len_k]

        """
        # [num_heads*batch_size, len_q, len_k]
        attns = torch.bmm(q, k.transpose(1, 2))
        attns = attns / self.temperature

        if mask is not None:
            attns.data.masked_fill(mask, -float('inf'))
            #  attns = attns.masked_fill(mask, -np.inf)
            #  attns.data.masked_fill_(1 - mask, -float('inf'))

        # [num_heads*batch_size, len_q, len_k]
        attns = self.softmax(attns)
        attns = self.dropout(attns)

        # [num_heads*batch_size, len_q, v_size]
        output = torch.bmm(attns, v)

        return output, attns

