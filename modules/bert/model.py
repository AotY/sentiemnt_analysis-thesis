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

        if self.model_type in ['bert_max_kernel', 'bert_avg_kernel']:
            self.linear_final = nn.Linear(config.embedding_size * 5, config.n_classes)
        elif self.model_type == 'bert_max_min':
            self.linear_final = nn.Linear(config.embedding_size * 2, config.n_classes)
        elif self.model_type.find('bert_rnn') != -1:
            self.rnn = RNNEncoder(config)
            self.linear_final = nn.Linear(config.hidden_size, config.n_classes)
        elif self.model_type.find('bert_cnn') != -1:
            self.cnn = CNNEncoder(config)
            self.linear_final = nn.Linear(len(config.kernel_heights) * config.out_channels, config.n_classes)
        elif self.model_type == 'bert_conv1d':
            self.conv1d1 = nn.Conv1d(config.embedding_size, config.embedding_size, 3) # 48
            self.max_pool1d1 = nn.MaxPool1d(3) # 16
            self.conv1d2 = nn.Conv1d(config.embedding_size, config.embedding_size, 2) # 15
            self.max_pool1d2 = nn.MaxPool1d(3) # 5
            self.conv1d3 = nn.Conv1d(config.embedding_size, config.embedding_size, 3) # 3
            self.max_pool1d3 = nn.MaxPool1d(3) # 1
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
        elif self.model_type == 'bert_min':
            outputs = outputs.transpose(1, 2)
            outputs = outputs.min(dim=2)[0]
        elif self.model_type == 'bert_max_min':
            outputs = outputs.transpose(1, 2)
            min_outputs = outputs.min(dim=2)[0]
            max_outputs = outputs.max(dim=2)[0]
            outputs = torch.cat((max_outputs, min_outputs), dim=1)
        elif self.model_type == 'bert_max_kernel':
            # [batch_size, embedding_size, max_len]
            outputs = outputs.transpose(1, 2)
            outputs = F.max_pool1d(outputs, kernel_size=10).squeeze(2)
            outputs = outputs.view(outputs.size(0), -1)
        elif self.model_type == 'bert_avg':
            outputs = outputs.transpose(1, 2)
            outputs = F.avg_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
            # outputs = outputs.mean(dim=1)
        elif self.model_type == 'bert_avg_kernel':
            # [batch_size, embedding_size, max_len]
            outputs = outputs.transpose(1, 2)
            outputs = F.avg_pool1d(outputs, kernel_size=10)
            outputs = outputs.view(outputs.size(0), -1)
        elif self.model_type.find('bert_rnn') != -1:
            # [max_len, batch_size, embedding_size]
            outputs = outputs.transpose(0, 1)
            # [batch_size, hidden_size]
            outputs, _ = self.rnn(outputs, lengths=lengths)
        elif self.model_type == 'bert_cnn':
            # [batch_size, embedding_size]
            outputs, _ = self.cnn(outputs)
        elif self.model_type == 'bert_conv1d':
            outputs = self.max_pool1d1(F.relu(self.conv1d1(outputs)))
            outputs = self.max_pool1d2(F.relu(self.conv1d2(outputs)))
            outputs = self.max_pool1d3(F.relu(self.conv1d3(outputs)))
            outputs = outputs.view(outputs.size(0), -1)

        if self.problem == 'classification':
            # [batch_size, ] -> [batch_size, n_classes]
            outputs = self.linear_final(outputs)

        return outputs, attns_list
