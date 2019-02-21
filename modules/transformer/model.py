#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer
"""
import torch
import torch.nn as nn
import torch.nn.funtional as F

from modules.transformer.encoder import Encoder
from modules.utils import rnn_factory


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, config, embedding):
        super().__init__()

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.model_type = config.model_type
        self.n_classes = config.n_classes
        self.transformer_size = config.transformer_size
        self.max_len = config.max_len

        self.encoder = Encoder(
            config,
            embedding
        )

        if self.model_type == 'transformer':
            self.linear_final = nn.Linear(
                self.transformer_size * self.max_len,
                self.n_classes,
            )
        elif self.model_type == 'transformer_mean':
            self.linear_final = nn.Linear(
                self.transformer_size,
                self.n_classes,
            )

        elif self.model_type == 'transformer_rnn':
            self.bidirection_num = 2 if config.bidirectional else 1
            self.hidden_size = self.transformer_size // self.bidirection_num

            # rnn
            self.rnn = rnn_factory(
                rnn_type=config.rnn_type,
                input_size=self.transformer_size,
                hidden_size=self.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout
            )

            self.linear_final = nn.Linear(
                self.transformer_size,
                self.n_classes,
            )
        elif self.model_type == 'transformer_weight':
            # W_s1
            self.linear_first = torch.nn.Linear(
                self.transformer_size, config.dense_size)
            self.linear_first.bias.data.fill_(0)

            # W_s2
            self.linear_second = torch.nn.Linear(
                config.dense_size, config.heads)
            self.linear_second.bias.data.fill_(0)

            self.linear_final = nn.Linear(
                config.max_len * config.heads, self.n_classes)

        nn.init.xavier_normal_(self.linear_final.weight)

    def forward(self,
                inputs,
                inputs_pos):
        """
        Args:
            inputs: [batch_size, max_len]
            inputs_pos: [batch_size, max_len]
        return: [batch_size * max_len, vocab_size]
        """
        # [batch_size, max_len, transformer_size]
        outputs, attns = self.encoder(inputs, inputs_pos, return_attns=True)

        # to [batch_size, n_classes]
        if self.model_type == 'transformer':
            # [batch_size, max_len * transformer_size]
            outputs = outputs.view(outputs.size(0), -1)
            outputs = F.log_softmax(self.linear_final(outputs), dim=1)
        elif self.model_type == 'transformer_mean':
            outputs = outputs.mean(1)
            outputs = F.log_softmax(self.linear_final(outputs), dim=1)
        elif self.model_type == 'transformer_rnn':
            outputs, _ = self.rnn(outputs.transpose(0, 1))
            outputs = self.rnn(outputs)[-1]
            outputs = F.log_softmax(self.linear_final(outputs), dim=1)
        elif self.model_type == 'transformer_weight':
            # [batch_size, max_len, dense_size]
            x = F.tanh(self.linear_first(outputs))

            # [batch_size, max_len, heads]
            x = self.linear_second(outputs)

            # [batch_size, n_classes]
            outputs = F.log_softmax(
                self.linear_final(x.view(x.size(0), -1)), dim=1)

        # [batch_size, vocab_size]
        return outputs, attns
