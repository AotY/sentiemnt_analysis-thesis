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
form modules.cnn_encoder import CNNEncoder
from modules.self_attention.model import StructuredSelfAttention
from modules.transformer.model import Transformer

from modules.utils import load_embeddings


class SAModel(nn.Module):
    '''
    generating responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''
    def __init__(self,
                 config,
                 pretrained_embedding=None,
                 device='cuda'):
        super(SAModel, self).__init__()

        self.config = config
        self.device = device

        embedding = load_embeddings(
            config,
            pretrained_embedding
        )

        if config.model_type == 'rnn':
            self.encoder = RNNEncoder(
                config,
                embedding
            )
        elif config.model_type == 'cnn':
            self.encoder = CNNEncoder(
                config,
                embedding
            )
        elif config.model_type == 'self_attention':
            self.encoder = StructuredSelfAttention(
                config,
                embedding
            )
        #  elif config.model_type in ['transformer', 'transformer_rnn', 'transformer_weight']:
        elif config.model.find('transformer') != -1:
            self.encoder = Transformer(
                config,
                embedding
            )

    def forward(self,
                inputs,
                lengths):
        '''
        Args:
            inputs: [q_max_len, batch_size]
            lengths: [batch_size]
        '''
        if self.config.model_type == 'rnn':
            # [batch_size, n_classes], None
            outputs, attns = self.encoder(
                inputs,
                lengths
            )
        elif config.model_type == 'cnn':
            # [batch_size, n_classes]
            outputs, attns = self.encoder(
                inputs,
                lengths
            )
        elif self.config.model_type == 'self_attention':
            # [batch_size, n_classes], [batch_size, heads, max_len]
            outputs, attns = self.encoder(
                inputs,
                lengths
            )
        #  elif config.model_type in ['transformer', 'transformer_rnn', 'transformer_weight']:
        elif self.config.model.find('transformer') != -1:
            # [batch_size, n_classes], [batch_size, max_len, max_len]
            outputs, attns = self.encoder(
                inputs,
                lengths
            )

        return outputs, attns
