#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Sentiment Analysis Model.
"""

import torch
import torch.nn as nn

from modules.rnn_encoder import RNNEncoder
from modules.cnn_encoder import CNNEncoder
from modules.rcnn_encoder import RCNNEncoder
from modules.self_attention.model import StructuredSelfAttention
from modules.transformer.model import TransformerCM
from modules.bert.model import BERTCM

from modules.utils import load_embedding


class SAModel(nn.Module):
    '''
    generating responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 config,
                 device='cuda',
                 pretrained_embedding=None):
        super(SAModel, self).__init__()

        self.config = config
        self.device = device
        # print('config: {}'.format(config))

        embedding = load_embedding(config, pretrained_embedding)

        #  if config.model_type in ['rnn', 'rnn_attention']:
        if self.config.model_type.startswith('rnn'):
            self.encoder = RNNEncoder(config, embedding)
        elif config.model_type == 'cnn':
            self.encoder = CNNEncoder(config, embedding)
        elif config.model_type == 'rcnn':
            self.encoder = RCNNEncoder(config, embedding)
        elif config.model_type == 'self_attention':
            self.encoder = StructuredSelfAttention(config, embedding)
        elif self.config.model_type.startswith('transformer'):
        #  elif config.model_type.find('transformer') != -1:
            self.encoder = TransformerCM(config, embedding)
        elif self.config.model_type.startswith('bert'):
        #  elif config.model_type.find('bert') != -1:
            self.encoder = BERTCM(config, embedding)

    def forward(self,
                inputs,
                lengths=None,
                inputs_pos=None):
        '''
        Args:
            inputs: [max_len, batch_size]
            inputs_pos: [max_len, batch_size]
            lengths: [batch_size]
        '''
        #  if self.config.model_type in ['rnn', 'rnn_attention']:
        if self.config.model_type.startswith('rnn'):
            # [batch_size, n_classes], None
            outputs, attns = self.encoder(
                inputs,
                lengths
            )
        elif self.config.model_type == 'cnn':
            # [batch_size, n_classes]
            outputs, attns = self.encoder(
                inputs.transpose(0, 1),
                lengths
            )
        elif self.config.model_type == 'rcnn':
            # [batch_size, n_classes], None
            outputs, attns = self.encoder(
                inputs,
                lengths
            )
        elif self.config.model_type == 'self_attention':
            # [batch_size, n_classes], [batch_size, num_heads, max_len]
            outputs, attns = self.encoder(
                inputs,
                lengths
            )
        #  elif self.config.model_type.find('transformer') != -1:
        elif self.config.model_type.startswith('transformer'):
            # [batch_size, n_classes], [num_heads * batch_size, max_len, max_len] list
            # print(inputs_pos)
            outputs, attns = self.encoder(
                inputs.transpose(0, 1),
                inputs_pos.transpose(0, 1),
                lengths
            )
        #  elif self.config.model_type.find('bert') != -1:
        elif self.config.model_type.startswith('bert'):
            # [batch_size, n_classes], [num_heads * batch_size, max_len, max_len] list
            # print(inputs_pos)
            outputs, attns = self.encoder(
                inputs.transpose(0, 1),
                inputs_pos.transpose(0, 1),
                lengths
            )
        else:
            raise ValueError('%s is invalid.' % self.config.model_type)

        return outputs, attns

    def test(self, input, input_pos):
        pass
