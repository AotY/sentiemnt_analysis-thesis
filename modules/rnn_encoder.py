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
from modules.cnn_encoder import CNNEncoder
from modules.bert.bert import BERT
from modules.reduce_state import ReduceState

from misc.vocab import PAD_ID


class RNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding=None):
        super(RNNEncoder, self).__init__()

        self.problem = config.problem

        self.model_type = config.model_type
        self.rnn_type = config.rnn_type

        # embedding
        if embedding is not None:
            self.embedding = embedding
            self.embedding_size = embedding.embedding_dim
            self.from_other = False
        else:
            self.from_other = True

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num
        self.n_classes = config.n_classes
        self.num_layers = config.num_layers

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = rnn_factory(
            rnn_type=self.rnn_type,
            input_size=config.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=config.bidirectional,
            dropout=0.5 if self.num_layers > 1 else 0
            # dropout=config.dropout if self.num_layers > 1 else 0
        )

        rnn_init(self.rnn_type, self.rnn)

        self.reduce_state = ReduceState(self.rnn_type)

        if self.model_type == 'rnn_bert':
            self.bert = BERT(config, None)

        if not self.from_other:
            if self.problem == 'classification':
                if self.model_type == 'rnn_cnn':
                    self.cnn = CNNEncoder(config)
                    self.linear_final = nn.Linear(
                        len(config.kernel_heights) * config.out_channels, config.n_classes)
                else:
                    self.linear_final = nn.Linear(
                        self.hidden_size * self.bidirection_num, self.n_classes)
            else:
                # self.linear_regression_dense = nn.Linear(
                    # self.hidden_size * self.bidirection_num, config.regression_dense_size)
                # self.linear_regression_final = nn.Linear(config.regression_dense_size, 1)
                self.linear_regression_final = nn.Linear(
                    self.hidden_size * self.bidirection_num, 1)

    def forward(self, inputs, lengths=None, hidden_state=None, inputs_pos=None):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [batch_size, n_classes]
        '''
        # embedded
        if not self.from_other:
            embedded = self.embedding(inputs)
            embedded = self.dropout(embedded)
        else:
            embedded = inputs

        # embedded: [max_len, batch_size, embedding_size]
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        if hidden_state is not None:
            outputs, hidden_state = self.rnn(embedded, hidden_state)
        else:
            outputs, hidden_state = self.rnn(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # print('outputs shape: ', outputs.shape)

        attns = None
        if self.model_type.find('attention') != -1:
            # [batch_size, max_len, hidden_size]
            outputs = outputs.permute(1, 0, 2)
            final_state = hidden_state
            if self.rnn_type == 'LSTM':
                final_state = hidden_state[0]
                final_state = final_state.view(
                    self.num_layers, final_state.size(1), -1)
                final_state = torch.sum(final_state, dim=0)
            outputs, attns = self.attention_net(outputs, final_state)
        elif self.model_type == 'rnn_cnn':
            # [batch_size, max_len, hidden_size]
            outputs = outputs.permute(1, 0, 2)
            outputs = self.cnn(outputs)
        elif self.model_type == 'rnn_avg':
            outputs = outputs.transpose(1, 2)
            outputs = F.avg_pool1d(
                outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'rnn_max':
            outputs = outputs.transpose(1, 2)
            outputs = F.max_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'rnn_bert':
            # [batch_size, max_len, hidden_size]
            outputs = outputs.permute(1, 0, 2)
            #  outputs = self.bert(outputs)
            outputs, attns = self.bert(outputs, None)
            # outputs = outputs[:, 0]
            outputs = outputs.mean(dim=1)
        else:
            # outputs = outputs[-1]
            hidden_state = self.reduce_state(hidden_state)
            if self.rnn_type == 'LSTM':
                hidden_state = hidden_state[0]
            # if hidden_state.size(0) > 1:
            outputs = hidden_state[-1]

        if not self.from_other:
            if self.problem == 'classification':
                # last step output [batch_size, hidden_state]
                outputs = self.linear_final(outputs)
            else:
                # outputs = self.linear_regression_dense(outputs)
                outputs = self.linear_regression_final(outputs)

        return outputs, attns

    def attention_net(self, lstm_outputs, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_outputs : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_outputs and and then finally computing the
                          new hidden state.

        Tensor Size :
            hidden.size() = (batch_size, hidden_size)
            attn_weights.size() = (batch_size, num_seq)
            soft_attn_weights.size() = (batch_size, num_seq)
            new_hidden_state.size() = (batch_size, hidden_size)
        """
        # print('fianl_state: ', final_state.shape)
        # [batch_size, hidden_size, 1]
        hidden = final_state.unsqueeze(2)
        # print('hidden: ', hidden.shape)

        # [batch_size, max_len, 1]
        attn_weights = torch.bmm(lstm_outputs, hidden)

        # [batch_size, max_len, 1]
        soft_attn_weights = F.softmax(attn_weights, 1)

        # [batch_size, hidden_size]
        new_hidden_state = torch.bmm(lstm_outputs.transpose(
            1, 2), soft_attn_weights).squeeze(2)

        return new_hidden_state, soft_attn_weights
