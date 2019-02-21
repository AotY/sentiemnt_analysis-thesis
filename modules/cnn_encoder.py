#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
CNN Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import init_linear_wt


class CNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding):
        super(CNNEncoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.dropout = nn.Dropout(config.dropout)

        self.conv2ds = nn.ModuleList()

        for channel, kernel_size, stride in zip(config.output_channels, config.kernel_sizes, config.strides):
            self.conv2ds.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    stride=stride
                )
            )

        self.maxpool2d = nn.MaxPool2d(kernel_size=config.maxpool_kernel_size)

        self.out_linear = nn.Linear(config.output_channels[-1], config.hidden_size)
        init_linear_wt(self.out_linear)

        self.linear_final = nn.Linear(config.hidden_size, config.n_classes)

    def forward(self, inputs, lengths=None, sort=False):
        """
        Args:
            inputs: [batch_size, max_len]
        Return:
            [batch_size, n_classes]
        """
        embedded = self.embedding(inputs)  # [batch_size, max_len, embedding_size]
        embedded = self.dropout(embedded)

        # [batch_size, 1, max_len, embedding_size]
        outputs = embedded.unsqueeze(1)

        # conv
        for conv2d in self.conv2ds:
            outputs = conv2d(outputs)
            outputs = F.relu(outputs)
            #  print('outputs: ', outputs.shape)
            outputs = outputs.transpose(1, 3)

        # [batch_size, 1, 1, 1024]
        outputs = self.maxpool2d(outputs)

        outputs = outputs.squeeze(2).transpose(0, 1)
        outputs = self.out_linear(outputs)  # [1, batch_size, hidden_size]

        outputs = F.log_softmax(self.linear_final(outputs[0]), dim=1)

        return outputs, None
