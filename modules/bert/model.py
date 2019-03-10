#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Bert classification model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import BERT

from modules.rnn_encoder import RNNEncoder
from modules.cnn_encoder import CNNEncoder


class BERTCM(nn.Module):
    """
    n-class classification model :
    """

    def __init__(self, config, embedding):
        """
        :param hidden: BERT model output size
        """
        super().__init__()

        self.problem = config.problem

        self.bert = BERT(config, embedding)

        self.model_type = config.model_type

        if self.model_type.find('bert_rnn') != -1:
            self.rnn = RNNEncoder(config)
            self.linear_final = nn.Linear(config.hidden_size, config.n_classes)
        elif self.model_type.find('bert_cnn') != -1:
            self.cnn = CNNEncoder(config)
            self.linear_final = nn.Linear(len(config.kernel_heights) * config.out_channels, config.n_classes)
        else:
            self.linear_final = nn.Linear(config.embedding_size, config.n_classes)

    def forward(self, inputs, inputs_pos, lengths=None):
        # [batch_size, max_len, embedding_size], [] list
        outputs, attns_list = self.bert(inputs, inputs_pos)

        if self.model_type == 'bert':
            outputs = outputs[:, 0]
        elif self.model_type == 'bert_max':
            outputs = outputs.transpose(1, 2)
            outputs = F.max_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
            # outputs = outputs.max(dim=1)[0]
        elif self.model_type == 'bert_avg':
            outputs = outputs.transpose(1, 2)
            outputs = F.avg_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
            # outputs = outputs.mean(dim=1)
        elif self.model_type.find('bert_rnn') != -1:
            # [max_len, batch_size, embedding_size]
            outputs = outputs.transpose(0, 1)
            # [batch_size, hidden_size]
            outputs, _ = self.rnn(outputs, lengths=lengths)
        elif self.model_type == 'bert_cnn':
            # [batch_size, embedding_size]
            outputs, _ = self.cnn(outputs)

        if self.problem == 'classification':
            # [batch_size, ] -> [batch_size, n_classes]
            outputs = self.linear_final(outputs)

        return outputs, attns_list
