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
import numpy as np
from .bert import BERT
from .utils.layer_norm import LayerNorm

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
        self.embedding_size = config.embedding_size
        self.batch_size = config.batch_size

        self.bert = BERT(config, embedding)

        self.model_type = config.model_type
        self.tau = config.tau

        self.norm = LayerNorm(self.embedding_size)
        self.linear_final = None
        if self.model_type in ['bert_max_kernel', 'bert_avg_kernel']:
            self.linear_final = nn.Linear(config.embedding_size * 5, config.n_classes)
        elif self.model_type == 'bert_max_min':
            self.linear_final = nn.Linear(config.embedding_size * 2, config.n_classes)
        elif self.model_type.find('bert_rnn') != -1:
            self.rnn = RNNEncoder(config)
            self.linear_final = nn.Linear(config.hidden_size, config.n_classes)
        elif self.model_type == 'bert_cnn':
            self.cnn = CNNEncoder(config)
            self.linear_final = nn.Linear(len(config.kernel_heights) * config.out_channels, config.n_classes)
        elif self.model_type == 'bert_sum':
            # self.norm = LayerNorm(config.embedding_size)
            # self.linear_dense = nn.Linear(config.embedding_size, config.embedding_size // 2)
            # self.linear_final = nn.Linear(config.embedding_size // 2, config.n_classes)
            self.linear_final = nn.Linear(config.embedding_size, config.n_classes)
        elif self.model_type == 'bert_weight':
            self.weight = nn.Parameter(torch.zeros(config.batch_size, config.max_len, 1, device=config.device))
            self.weight.data.fill_(0)
            # self.weight.data.uniform_(-np.sqrt(3.0 / config.embedding_size), np.sqrt(3.0 / config.embedding_size))
        elif self.model_type == 'bert_weight_embedding':
            self.weight = nn.Parameter(torch.zeros(config.batch_size, config.embedding_size, 1, device=config.device))
            # self.weight.data.fill_(0)
            self.weight.data.uniform_(-np.sqrt(3.0 / config.embedding_size), np.sqrt(3.0 / config.embedding_size))
        elif self.model_type == 'bert_sample_manual':
            self.sample_weight = torch.ones(config.max_len) / config.max_len
            self.sample_num = 6
            self.sample_weight[:20] = 0.1
            self.sample_weight[:10] = 0.2
            self.sample_weight[:5] = 0.3
            self.linear_final = nn.Linear(config.embedding_size * self.sample_num, config.n_classes)
        elif self.model_type == 'bert_sample_exp':
            self.linear_final = nn.Linear(config.embedding_size, config.n_classes)
        elif self.model_type.startswith('bert_gumbel'):
            self.linear_final = nn.Linear(config.embedding_size, config.n_classes)
        elif self.model_type == 'bert_conv1d':
            self.conv1d1 = nn.Conv1d(config.embedding_size, config.embedding_size, 6) # 45
            self.max_pool1d1 = nn.MaxPool1d(3) # 15
            self.conv1d2 = nn.Conv1d(config.embedding_size, config.embedding_size, 4) # 12
            self.max_pool1d2 = nn.MaxPool1d(2) # 6
            self.conv1d3 = nn.Conv1d(config.embedding_size, config.embedding_size, 4) # 3
            self.max_pool1d3 = nn.MaxPool1d(3) # 1
            # self.linear_dense = nn.Linear(config.embedding_size, config.embedding_size // 2)
            # self.dropout_dense = nn.Dropout(0.5)
            # self.linear_final = nn.Linear(config.embedding_size // 2, config.n_classes)
            # self.norm = LayerNorm(config.embedding_size)
            self.linear_final = nn.Linear(config.embedding_size, config.n_classes)
        if self.linear_final is None:
            if self.model_type in ['bert_max_embedding', 'bert_avg_embedding', 'bert_weight_embedding']:
                self.linear_final = nn.Linear(config.max_len, config.n_classes)
            else:
                self.linear_final = nn.Linear(config.embedding_size, config.n_classes)

        if self.model_type.startswith('bert_weight') or self.model_type.startswith('bert_gumbel'):
            self.position_weights = None

    def forward(self, inputs, inputs_pos, lengths=None):
        # [batch_size, max_len, embedding_size], [] list
        outputs, attns_list, embedded = self.bert(inputs, inputs_pos)

        if self.model_type == 'bert':
            outputs = outputs[:, 0]
        elif self.model_type == 'bert_sum':
            outputs = outputs.transpose(1, 2)
            outputs = outputs.sum(dim=2)
            outputs = self.norm(outputs)
            # outputs = self.linear_dense(outputs)
        elif self.model_type == 'bert_max':
            outputs = outputs.transpose(1, 2)
            outputs = F.max_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
            outputs = self.norm(outputs)
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
            #  outputs = self.norm(outputs)
        elif self.model_type == 'bert_avg_kernel':
            # [batch_size, embedding_size, max_len]
            outputs = outputs.transpose(1, 2)
            outputs = F.avg_pool1d(outputs, kernel_size=10)
            outputs = outputs.view(outputs.size(0), -1)
        elif self.model_type == 'bert_avg_embedding':
            outputs = F.avg_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'bert_max_embedding':
            outputs = F.max_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'bert_residual_embedding_avg':
            outputs = outputs + embedded
            outputs = self.norm(outputs)
            outputs = outputs.transpose(1, 2)
            outputs = F.avg_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'bert_residual_embedding_max':
            outputs = outputs + embedded
            outputs = self.norm(outputs)
            outputs = outputs.transpose(1, 2)
            outputs = F.max_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'bert_sample_manual':
            random_idxes = torch.multinomial(self.sample_weight, self.sample_num).to(outputs.device)
            outputs = outputs.index_select(1, random_idxes)
            outputs = outputs.view(outputs.size(0), -1)
        elif self.model_type == 'bert_sample_exp':
            # [batch_size, max_len]
            mean_outputs = outputs.mean(dim=2)
            indexs = torch.multinomial(torch.exp(mean_outputs), 1)  # batch_size x 1 (sampling from each row)
            # [ batch_size, embedding_size]
            outputs = torch.cat([torch.index_select(output, 0, index) for output, index in zip(outputs, indexs)])
        elif self.model_type == 'bert_gumbel_avg':
            outputs = outputs.permute(0, 2, 1)
            batch_size, embedding_size, _ = outputs.size()

            # [batch_size * embedding_size, max_len]
            outputs = outputs.contiguous().view(-1, outputs.size(2))

            # [batch_size * embedding_size, max_len]
            gumbel_outputs = F.gumbel_softmax(outputs, hard=True)

            # [batch_size * embedding_size, batch_size * embedding_size]
            weight_outputs = torch.matmul(outputs, gumbel_outputs.transpose(0, 1))

            # [batch_size, embedding_size, batch_size * embedding_size]
            outputs = weight_outputs.contiguous().view(batch_size, embedding_size, weight_outputs.size(1))

            # [batch_size, embedding_size]
            outputs = F.avg_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
        elif self.model_type == 'bert_gumbel_sum':
            outputs = outputs.permute(0, 2, 1)
            batch_size, embedding_size, _ = outputs.size()

            # [batch_size * embedding_size, max_len]
            outputs = outputs.contiguous().view(-1, outputs.size(2))

            # [batch_size * embedding_size, max_len]
            gumbel_outputs = F.gumbel_softmax(outputs, hard=True)

            # [batch_size * embedding_size, batch_size * embedding_size]
            weight_outputs = torch.matmul(outputs, gumbel_outputs.transpose(0, 1))

            # [batch_size, embedding_size, batch_size * embedding_size]
            outputs = weight_outputs.contiguous().view(batch_size, embedding_size, weight_outputs.size(1))

            # [batch_size, embedding_size]
            outputs = outputs.sum(dim=2)
            outputs = self.norm(outputs)
        elif self.model_type == 'bert_gumbel_tau':
            outputs = outputs.permute(0, 2, 1)
            batch_size, embedding_size, _ = outputs.size()

            # [batch_size * embedding_size, max_len]
            outputs = outputs.contiguous().view(-1, outputs.size(2))

            # [batch_size * embedding_size, max_len]
            gumbel_outputs = F.gumbel_softmax(outputs, tau=self.tau)

            # [batch_size * embedding_size, batch_size * embedding_size]
            weight_outputs = torch.matmul(outputs, gumbel_outputs.transpose(0, 1))

            # [batch_size, embedding_size, batch_size * embedding_size]
            outputs = weight_outputs.contiguous().view(batch_size, embedding_size, weight_outputs.size(1))

            # [batch_size, embedding_size]
            outputs = F.avg_pool1d(outputs, kernel_size=outputs.size(2)).squeeze(2)
            #  outputs = torch.sum(outputs, dim=2)
            # save gumbel_outputs, for visualization
            position_weights = gumbel_outputs.numpy()  # [batch_size * embedding_size, max_len]
        elif self.model_type.find('bert_rnn') != -1:
            # [max_len, batch_size, embedding_size]
            outputs = outputs.transpose(0, 1)
            # [batch_size, hidden_size]
            outputs, _ = self.rnn(outputs, lengths=lengths)
        elif self.model_type == 'bert_cnn':
            # [batch_size, embedding_size]
            outputs, _ = self.cnn(outputs)
        elif self.model_type == 'bert_weight':
            # [batch_size, embedding_size, max_len]
            outputs = outputs.transpose(1, 2)
            outputs = outputs @ self.weight
            outputs = outputs.view(outputs.size(0), -1)

            #  position_weights = self.weight.view(-1).numpy()  # [batch_size * embedding_size, max_len]
            position_weights = self.weight.squeeze(2).numpy()  # [batch_size * embedding_size, max_len]
        elif self.model_type == 'bert_weight_embedding':
            # [batch_size, max_len, embedding_size]
            outputs = outputs @ self.weight
            outputs = outputs.view(outputs.size(0), -1)
        elif self.model_type == 'bert_conv1d':
            outputs = outputs.transpose(1, 2)
            outputs = self.max_pool1d1(F.relu(self.conv1d1(outputs)))
            outputs = self.max_pool1d2(F.relu(self.conv1d2(outputs)))
            outputs = self.max_pool1d3(F.relu(self.conv1d3(outputs)))
            # print('outputs: ', outputs.shape)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = self.norm(outputs)
            # outputs = self.linear_dense(outputs)
            # outputs = self.dropout_dense(outputs)

        if self.problem == 'classification':
            # [batch_size, ] -> [batch_size, n_classes]
            outputs = self.linear_final(outputs)

        if self.model_type.startswith('bert_weight') or self.model_type.startswith('bert_gumbel'):
            if self.position_weights is None:
                self.position_weights = position_weights
            else:
                self.position_weights += position_weights
        return outputs, attns_list
