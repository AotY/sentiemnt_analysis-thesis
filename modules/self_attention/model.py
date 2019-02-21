#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
self-attention
https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import rnn_factory, rnn_init
from misc.vocab import PAD_ID


class StructuredSelfAttention(nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """

    def __init__(self,
                 config,
                 embedding):
        """
        Initializes parameters suggested in paper

        Args:
            batch_size  : {int} batch_size used for training
            hidden_size: {int} hidden dimension for lstm
            dense_size         : {int} hidden dimension for the dense layer
            num_heads           : {int} attention-hops or attention num_heads
            max_len     : {int} number of lstm timesteps
            embedding_size     : {int} embedding dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embedding
            embedding  : {torch.FloatTensor} loaded pretrained embedding
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.n_classes = config.n_classes
        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.num_heads = config.num_heads
        self.dense_size = config.dense_size

        # dropout
        #  self.dropout = nn.Dropout(config.dropout)

        #  self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, config.num_layers)
        self.rnn = rnn_factory(
            rnn_type=config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )

        rnn_init(config.rnn_type, self.rnn)

        # W_s1
        self.linear_first = torch.nn.Linear(self.hidden_size, self.dense_size)
        self.linear_first.bias.data.fill_(0)

        # W_s2
        self.linear_second = torch.nn.Linear(self.dense_size, self.num_heads)
        self.linear_second.bias.data.fill_(0)

        self.linear_final = nn.Linear(self.hidden_size, self.n_classes)

    def init_hidden(self, config):
        return (torch.zeros(1, self.batch_size, self.hidden_size), torch.zeros(1, self.batch_size, self.hidden_size))

    def forward(self, inputs, lengths=None, hidden_state=None):
        """
        inputs: [max_len, batch_size]
        """
        embedded = self.embedding(inputs)
        #  embedded = self.dropout(embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        # [max_len, batch_size, hidden_size]
        total_length = inputs.size(0)
        if hidden_state is None:
            outputs, hidden_state = self.rnn(embedded)
        else:
            outputs, hidden_state = self.rnn(embedded, hidden_state)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, padding_value=PAD_ID, total_length=total_length)

        # [batch_size, max_len, hidden_size]
        x = outputs.transpose(0, 1)

        # [batch_size, max_len, dense_size]
        x = F.tanh(self.linear_first(outputs))

        # [batch_size, max_len, num_heads]
        x = self.linear_second(outputs)

        # [batch_size, max_len, num_heads]
        x = F.softmax(outputs, dim=1)

        # [batch_size, num_heads, max_len]
        attns = x.transpose(1, 2)

        # [batch_size, num_heads, hidden_size]
        sentence_embeddings = attns @ outputs.transpose(0, 1)

        # [batch_size, hidden_size]
        avg_sentence_embeddings = torch.sum(sentence_embeddings, dim=1) / self.num_heads

        # [batch_size, n_classes]
        outputs = F.log_softmax(self.linear_final(avg_sentence_embeddings), dim=1)

        return outputs, attns

        #  if not bool(self.type):
            #  output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
            #  return output,attention
        #  else:

	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation

        Args:
           m: {Variable} ||AAT - I||

        Returns:
            regularized value

        """
        return torch.sum(torch.sum(torch.sum(m**2, 1), 1) ** 0.5).type(torch.DoubleTensor)

