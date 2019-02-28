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


class RCNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding):
        super(RCNNEncoder, self).__init__()

        self.model_type = config.model_type
        self.rnn_type = config.rnn_type

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

        self.W2 = nn.Linear(self.hidden_size * self.bidirection_num + self.embedding_size, self.hidden_size)

        if self.problem == 'classification':
            self.linear_final = nn.Linear(self.hidden_size, self.n_classes)
        else:
            self.linear_regression_dense = nn.Linear(self.hidden_size, config.regression_dense_size)
            self.linear_regression_final = nn.Linear(config.regression_dense_size, 1)

    def forward(self, inputs, lengths=None, hidden_state=None):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [batch_size, n_classes]
        '''
        # embedded
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        print('embedded shape: ', embedded.shape)
        if lengths is not None:
            rnn_inputs = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        if hidden_state is not None:
            outputs, hidden_state = self.rnn(rnn_inputs, hidden_state)
        else:
            outputs, hidden_state = self.rnn(rnn_inputs)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, padding_value=PAD_ID, total_length=embedded.size(0))
        print('outputs shape: ', outputs.shape)

        # [batch_size, max_len, hidden_size + embedding_size]
        outputs = torch.cat((outputs, embedded), dim=2).permute(1, 0, 2)

        # [batch_size, max_len, hidden_size]
        outputs = self.W2(outputs)

        # [batch_size, hidden_size, max_len]
        outputs = outputs.permute(0, 2, 1)

        # [batch_size, hidden_size, 1]
        outputs = F.max_pool1d(outputs, outputs.size(2))
        print('outputs shape: ', outputs.shape)

        # [batch_size, hidden_size]
        outputs = outputs.squeeze(2)

        if self.problem == 'classification':
            outputs = self.linear_final(outputs)
        else:
            outputs = self.linear_regression_dense(outputs)
            outputs = self.linear_regression_final(outputs)
        print('outputs shape: ', outputs.shape)

        return outputs, None

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final outputs of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                          new hidden state.

        Tensor Size :
            hidden.size() = (batch_size, hidden_size)
            attn_weights.size() = (batch_size, num_seq)
            soft_attn_weights.size() = (batch_size, num_seq)
            new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state
