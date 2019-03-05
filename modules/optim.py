#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.
"""
custom optim
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/optim/optim.py
"""

import torch
import torch.nn as nn
import numpy as np


"""
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
"""
class ScheduledOptimizer:
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, embedding_size, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(embedding_size, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step'''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
