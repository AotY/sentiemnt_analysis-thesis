#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/bert.py
"""

import torch.nn as nn

from .transformer_block import TransformerBlock
#  from .embedding import PositionalEmbedding

from misc.vocab import PAD_ID


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self,
                 config,
                 embedding=None):
                 #  vocab_size, d_model=768, num_layers=12, num_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param d_model: BERT model d_model size
        :param num_layers: numbers of Transformer blocks(layers)
        :param num_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.max_len = config.max_len

        self.use_pos = config.use_pos

        self.d_model = config.embedding_size
        self.num_layers = config.t_num_layers
        self.num_heads = config.num_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        #  self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)
        if embedding is not None:
            self.embedding = embedding
            if self.use_pos:
                #  self.pos_embedding = PositionEmbedding(self.d_model, config.max_len)
                self.pos_embedding = nn.Embedding(
                    self.max_len + 1, self.d_model, padding_idx=PAD_ID)

            self.from_other = False
        else:
            self.from_other = True

        self.dropout = nn.Dropout(p=config.dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads,
                             self.d_model * 4, config.dropout)
            for _ in range(self.num_layers)
        ])

    def forward(self, inputs, inputs_pos=None):
        # attention masking for padded
        if not self.from_other:
            if self.use_pos:
                # embedding the indexed sequence to sequence of vectors
                w_embedded = self.embedding(inputs)
                p_embedded = self.pos_embedding(inputs_pos)
                #  print('w_embedded: ', w_embedded.dtype)
                #  print('p_embedded: ', p_embedded.dtype)
                embedded = w_embedded + p_embedded
            else:
                embedded = self.embedding(inputs)
            # torch.ByteTensor([batch_size, 1, max_len, max_len)
            mask = (inputs > 0).unsqueeze(1).repeat(1, inputs.size(1), 1).unsqueeze(1)
        else:
            embedded = inputs
            mask = None

        # running over multiple transformer blocks
        attns_list = list()
        outputs = embedded
        for transformer in self.transformer_blocks:
            outputs, attns = transformer.forward(outputs, mask)
            attns_list.append(attns)

        outputs = self.dropout(outputs)

        return outputs, attns_list, embedded
