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

        self.linear_final = nn.Linear(config.embedding_size, config.n_classes)

    def forward(self, inputs, inputs_pos):
        # [batch_size, max_len, embedding_size], [] list
        outputs, attns_list = self.bert(inputs, inputs_pos)
        if self.model_type == 'bert':
            outputs = outputs[:, 0]
        elif self.model_type == 'bert_max':
            outputs = outputs.max(dim=1)[0]
            pass
        elif self.model_type == 'bert_avg':
            outputs = outputs.mean(dim=1)
            pass

        if self.problem == 'classification':
            # [batch_size, n_classes]
            outputs = self.linear_final(outputs)

        return outputs, attns_list
