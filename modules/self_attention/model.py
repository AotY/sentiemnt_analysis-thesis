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

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num

        self.n_classes = config.n_classes
        self.num_heads = config.num_heads

        # dropout
        self.dropout = nn.Dropout(config.dropout)

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
        self.linear_first = torch.nn.Linear(self.hidden_size * self.bidirection_num, config.dense_size)
        self.linear_first.bias.data.fill_(0)

        # W_s2
        self.linear_second = torch.nn.Linear(config.dense_size, self.num_heads)
        self.linear_second.bias.data.fill_(0)

        self.linear_final = nn.Linear(self.hidden_size * self.bidirection_num, config.n_classes)

    def forward(self, inputs, lengths=None, hidden_state=None):
        """
        inputs: [max_len, batch_size]
        """
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        # [max_len, batch_size, hidden_size]
        if hidden_state is None:
            outputs, hidden_state = self.rnn(embedded)
        else:
            outputs, hidden_state = self.rnn(embedded, hidden_state)

        # [max_len, batch_size, hidden_size]
        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # [batch_size, max_len, dense_size]
        x = torch.tanh(self.linear_first(outputs.transpose(0, 1)))
        # print('x shape: ', x.shape)

        # [batch_size, max_len, num_heads]
        x = self.linear_second(x)
        # print('x shape: ', x.shape)

        # [batch_size, max_len, num_heads]
        x = F.softmax(x, dim=1)
        # print('x shape: ', x.shape)

        # [batch_size, num_heads, max_len]
        attns = x.transpose(1, 2)

        # print('attns shape: ', attns.shape)

        # [batch_size, num_heads, hidden_size]
        # Hidden states are weighted by an attention vector (A’s) to obtain a
        # refined sentence representation (M in the paper).
        sentence_embeddings = attns @ outputs.transpose(0, 1)

        # [batch_size, hidden_size]
        avg_sentence_embeddings = torch.sum(sentence_embeddings, dim=1) / self.num_heads

        # [batch_size, n_classes]
        outputs = self.linear_final(avg_sentence_embeddings)

        return outputs, attns

        #  if not bool(self.type):
            #  output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
            #  return output,attention
        #  else:

    def l2_matrix_norm(self, M):
        """
        M can suffer from redundancy problems if the attention mechanism always provides similar
        summation weights for all the r hops.

        We use the dot product of A and its transpose, subtracted by an identity matrix.

        This penalization term P will be multiplied by a coefficient, and we minimize it together
        with the original loss, which is dependent on the downstream application.

        In the most extreme case, where there is no overlap between the two probability
        distributions ai and aj, the correspond aij will be 0. Otherwise, it will have a
        positive value.

        On the other extreme end, if the two distribution are identical and all concentrates
        on one single word, it will have a maximum value of 1.

        We subtract an identity matrix from AAT so that forces the elements on the diagonal
        of AAT to approximate 1, which encourages each summation vector ai to focus on as
        few number of words as possible, forcing all other elements to 0, which punishes
        redundancy between different summation vectors.

        This penalty encourages the self-attention matrix to have large values on its diagonal
        and it lets single attention weights for a given token dominates other (r-1) attention weights.

        为了使每一个分布关注的地方不一样（每一行即是一个分布）, 对角线元素的最大值为1，其他元素的最小值为0,
        这个惩罚的目的是为了让对角线元素尽量接近1

        in order to make each distributions different (each row)

        Frobenius norm calculation, similar to an L2 regularization term.

        Args:
           M:  ||AAT - I||
        Returns:
            regularized value

        """
        return torch.sum(torch.sum(torch.sum(M**2, 1), 1) ** 0.5).type(torch.DoubleTensor)

