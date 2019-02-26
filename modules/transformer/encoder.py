#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer Encoder.
"""

import torch.nn as nn
from modules.transformer.layers import EncoderLayer
from modules.transformer.utils import get_sinusoid_encoding_table
from modules.transformer.utils import get_attn_key_pad_mask
from modules.transformer.utils import get_non_pad_mask

from misc.vocab import PAD_ID


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, config, embedding):
        super(Encoder, self).__init__()

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.use_pos = config.use_pos

        self.dropout = nn.Dropout(config.dropout)

        n_position = config.max_len + 1
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position,
                                        self.embedding_size,
                                        padid=PAD_ID),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.t_num_layers)]
        )

    def forward(self,
                enc_inputs,
                enc_inputs_pos=None,
                return_attns=True):
        """
        Args:
            enc_inputs: [batch_size, max_len]
            enc_inputs_pos: [batch_size, max_len]
        return:
            enc_outputs: [batch_size, max_len, embedding_size]
            enc_slf_attn_list: [
                [num_heads * batch_size, max_len, max_len],
            ]
        """
        # print('enc_inputs: ', enc_inputs.shape)
        # print('enc_inputs_pos: ', enc_inputs_pos.shape)
        if return_attns:
            enc_slf_attn_list = list()

        # -- Prepare masks
        attn_mask = get_attn_key_pad_mask(k=enc_inputs, q=enc_inputs, padid=PAD_ID)
        #  print('attn_mask: ', attn_mask)

        non_pad_mask = get_non_pad_mask(enc_inputs, PAD_ID)
        #  print('non_pad_mask: ', non_pad_mask)

        enc_embedded = self.embedding(enc_inputs) # [batch_size, max_len, embedding_size]

        # dropout, default: 0.1
        enc_embedded = self.dropout(enc_embedded)

        if self.use_pos:
            pos_embedded = self.pos_embedding(enc_inputs_pos).to(enc_inputs.device)
            # print('enc_embedded: ', enc_embedded.shape)
            # print('pos_embedded: ', pos_embedded.shape)
            enc_embedded = enc_embedded + pos_embedded

        enc_outputs = enc_embedded
        for layer in self.layer_stack:
            enc_outputs, en_slf_attn = layer(
                enc_outputs,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask
            )

            if return_attns:
                enc_slf_attn_list.append(en_slf_attn)

        # print('enc_outputs shape: ', enc_outputs.shape)
        if return_attns:
            return enc_outputs, enc_slf_attn_list

        # [batch_size, max_len, embedding_size]
        return enc_outputs
