#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.
"""
From Attention is All You Need.
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Layers.py
"""

import torch
import torch.nn as nn
from modules.transformer.sub_layers import MultiHeadAttention
from modules.transformer.sub_layers import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """compose with two layer."""

    def __init__(self,
                 config):
        super(EncoderLayer, self).__init__()

        self.mh_attn = MultiHeadAttention(
            config
        )

        self.pos_ffn = PositionwiseFeedForward(
            config
        )

    def forward(self,
                enc_input,
                non_pad_mask=None,
                attn_mask=None):
        """
        Args:
            enc_input: [batch_size, max_len, transformer_size]
            non_pad_mask: [batch_size, max_len, transformer_size]
            attn_mask: [batch_size, max_len, 1]
        """
        #  print('enc_input: ', enc_input.shape)
        #  print('non_pad_mask: ', non_pad_mask.shape)
        #  print('attn_mask: ', attn_mask.shape)

        enc_output, enc_attn = self.mh_attn(
            enc_input,
            enc_input,
            enc_input,
            mask=attn_mask
        )
        #  print('layer enc_output: ', enc_output)

        enc_output = enc_output * non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        enc_output = enc_output * non_pad_mask

        return enc_output, enc_attn


