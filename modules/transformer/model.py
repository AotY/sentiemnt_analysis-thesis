#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#  from modules.transformer.encoder import Encoder
from modules.utils import rnn_factory
from modules.transformer.layers import EncoderLayer
from modules.transformer.utils import get_sinusoid_encoding_table
from modules.transformer.utils import get_attn_key_pad_mask
from modules.transformer.utils import get_non_pad_mask

from misc.vocab import PAD_ID


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, config, embedding):
        super().__init__()

        self.problem = config.problem

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.model_type = config.model_type
        self.n_classes = config.n_classes
        self.dense_size = config.dense_size

        #  self.transformer_size = config.transformer_size
        self.max_len = config.max_len

        #  self.encoder = Encoder(config, embedding)
        self.use_pos = config.use_pos

        self.dropout = nn.Dropout(config.dropout)

        if self.use_pos:
            n_position = config.max_len + 1
            self.pos_embedding = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(n_position,
                                            self.embedding_size,
                                            padid=PAD_ID),
                freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.t_num_layers)]
        )

        if self.model_type == 'transformer':
            in_feature_size = self.embedding_size * self.max_len
        elif self.model_type == 'transformer_mean':
            # self.linear_dense = nn.Linear(
                # self.embedding_size,
                # self.dense_size
            # )
            in_feature_size = self.embedding_size

        elif self.model_type == 'transformer_rnn':
            self.bidirection_num = 2 if config.bidirectional else 1
            self.hidden_size = self.embedding_size // self.bidirection_num
            self.rnn = rnn_factory(
                rnn_type=config.rnn_type,
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout
            )
            in_feature_size = self.hidden_size * self.bidirection_num
        elif self.model_type == 'transformer_weight':
            # W_s1
            self.linear_first = torch.nn.Linear(self.embedding_size, config.dense_size)
            self.linear_first.bias.data.fill_(0)

            # W_s2
            self.linear_second = torch.nn.Linear(config.dense_size, config.num_heads)
            self.linear_second.bias.data.fill_(0)
            in_feature_size = config.max_len * config.num_heads
        elif self.model_type == 'transformer_maxpool':
            in_feature_size = self.embedding_size
        elif self.model_type == 'transformer_maxpool_concat': # and transformer_maxpool_residual
            self.W2 = nn.Linear(self.embedding_size * 2, self.embedding_size)
            in_feature_size = self.embedding_size
        elif self.model_type == 'transformer_maxpool_residual':
            #  self.layer_norm = nn.LayerNorm(config.embedding_size)
            in_feature_size = self.embedding_size

        if self.problem == 'classification':
            self.linear_final = nn.Linear(in_feature_size, config.n_classes)
            # nn.init.xavier_normal_(self.linear_final.weight)
        else:
            self.linear_regression_dense = nn.Linear(in_feature_size, config.regression_dense_size)
            self.linear_regression_final = nn.Linear(config.regression_dense_size, 1)


    def forward(self, inputs, inputs_pos):
        """
        Args:
            inputs: [batch_size, max_len]
            inputs_pos: [batch_size, max_len]
        return:
            outputs: [batch_size, n_classes]
            attns: [batch_size * num_heads, max_len, max_len] list
        """
        slf_attn_list = list()
        # [batch_size, ]
        attn_mask = get_attn_key_pad_mask(k=inputs, q=inputs, padid=PAD_ID)

        # [batch_size, ]
        non_pad_mask = get_non_pad_mask(inputs, PAD_ID)

        # [batch_size, max_len, embedding_size]
        embedded = self.embedding(inputs)

        embedded = self.dropout(embedded)

        if self.use_pos:
            pos_embedded = self.pos_embedding(enc_inputs_pos).to(inputs.device)
            embedded = embedded + pos_embedded

        if self.model_type.count('residual') != -1:
            residual = embedded

        outputs = embedded
        for layer in self.layer_stack:
            outputs, slf_attn = layer(
                outputs,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask
            )

            slf_attn_list.append(slf_attn)

        # [batch_size, max_len, embedding_size] list
        #  outputs, attns = self.encoder(inputs, inputs_pos, return_attns=True)

        # to [batch_size, n_classes]
        if self.model_type == 'transformer':
            # [batch_size, max_len * embedding_size]
            outputs = outputs.view(outputs.size(0), -1)
        elif self.model_type == 'transformer_mean':  # mean, average
            # [batch_size, embedding_size, max_len]
            outputs = outputs.permute(0, 2, 1)
            # [batch_size, embedding_size]
            outputs = outputs.mean(dim=2)
        elif self.model_type == 'transformer_rnn':  # with or without position embedding
            # [max_len, batch_size, hidden_size]
            outputs, _ = self.rnn(outputs.transpose(0, 1))
            outputs = outputs[-1]
        elif self.model_type == 'transformer_weight':
            # [batch_size, max_len, dense_size]
            outputs = F.tanh(self.linear_first(outputs))
            # [batch_size, max_len, num_heads]
            outputs = self.linear_second(outputs)
            # [batch_size, max_len * num_heads]
            outputs = x.view(x.size(0), -1)
        elif self.model_type == 'transformer_maxpool':
            # [batch_size, embedding_size, max_len]
            outputs = outputs.permute(0, 2, 1)
            # [batch_size, embedding_size]
            outputs = F.max_pool1d(outputs, outputs.size(2)).squeeze(2)
        elif self.model_type == 'transformer_maxpool_concat':
            # [batch_size, max_len, embedding_size * 2]
            outputs = torch.cat((outputs, residual), dim=2)
            # [batch_size, max_len, embedding_size]
            outputs = self.W2(outputs)
            # [batch_size, embedding_size, max_len]
            outputs = outputs.permute(0, 2, 1)
            # [batch_size, embedding_size]
            outputs = F.max_pool1d(outputs, outputs.size(2)).squeeze(2)
        elif self.model_type == 'transformer_maxpool_residual':
            # [batch_size, max_len, embedding_size]
            outputs = outputs + residual
            #  outputs = self.layer_norm(outputs + residual)

            # [batch_size, embedding_size, max_len]
            outputs = outputs.permute(0, 2, 1)
            # [batch_size, embedding_size]
            outputs = F.max_pool1d(outputs, outputs.size(2)).squeeze(2)

        if self.problem == 'classification':
            # [batch_size, n_classes]
            outputs = self.linear_final(outputs)
        else:
            outputs = self.linear_regression_dense(outputs)
            # [batch_size, 1]
            outputs = self.linear_regression_final(outputs)

        # [batch_size, vocab_size]
        return outputs, slf_attn_list
