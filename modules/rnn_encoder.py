#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
RNN Encoder
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import rnn_factory, rnn_init
from misc.vocab import PAD_ID


class RNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding):
        super(RNNEncoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num
        self.n_classes = config.n_classes

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = rnn_factory(
            rnn_type=config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )

        rnn_init(config.rnn_type, self.rnn)

        self.linear_final = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, inputs, lengths, hidden_state=None, sort=True):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch_size, num_directions * hidden_size]
        '''
        if lengths is None:
            raise ValueError('lengths is none.')

        total_length = inputs.size(0)

        sorted_lengths = lengths

        if not sort:
            # sort lengths
            sorted_lengths, sorted_indexes = torch.sort(
                lengths, dim=0, descending=True)

            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)

            # [max_len, batch_size]
            inputs = inputs.index_select(1, sorted_indexes)

        # embedded
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths)

        if hidden_state is not None:
            outputs, hidden_state = self.rnn(embedded, hidden_state)
        else:
            outputs, hidden_state = self.rnn(embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, padding_value=PAD_ID, total_length=total_length)

        if not sort:
            # [max_len, batch_size, hidden_state]
            outputs = outputs.index_select(1, restore_indexes).contiguous()

            # [num_layer * bidirection_num, batch_size, hidden_state / bidirection_num]
            hidden_state = hidden_state.index_select(
                1, restore_indexes).contiguous()

        # last step output [batch_size, hidden_state]
        output = outputs[-1]

        output = F.log_softmax(self.linear_final(output), dim=1)

        return output, None
