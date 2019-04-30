#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from misc.vocab import PAD_ID

"""
    Utils
"""

def load_embedding(config, pretrained_embedding=None):
    """Load the pretrained_embedding based on flag"""

    if config.use_pretrained_embedding is True and pretrained_embedding is None:
        raise Exception("Send a pretrained word embedding as an argument")

    if not config.use_pretrained_embedding and config.vocab_size is None:
        raise Exception("Vocab size cannot be empty")

    if not config.use_pretrained_embedding:
        embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=PAD_ID)

    elif config.use_pretrained_embedding:
        embedding = nn.Embedding(pretrained_embedding.size(0), pretrained_embedding.size(1), padding_idx=PAD_ID)
        embedding.weight = nn.Parameter(pretrained_embedding)

    return embedding

# get rnn by attr
def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available
    if rnn_type in ['RNN', 'LSTM', 'GRU']:
        return getattr(nn, rnn_type)(**kwargs)
    else:
        raise ValueError("%s is not valid." % rnn_type)

# init rnn
def rnn_init(rnn_type, rnn):
    if rnn_type == 'LSTM':
        init_lstm_orth(rnn)
    elif rnn_type == 'GRU':
        init_gru_orth(rnn)

# orthogonal initialization
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)

def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    #positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4 : l // 2].data.fill_(1.0)
        hh_b[l // 4 : l // 2].data.fill_(1.0)

def init_linear_wt(linear):
    init_wt_normal(linear.weight, linear.in_features)
    if linear.bias is not None:
        init_wt_normal(linear.bias, linear.in_features)

def init_wt_normal(weight, dim=512):
    weight.data.normal_(mean=0, std=np.sqrt(2.0 / dim))

def init_wt_unif(weight, dim=512):
    weight.data.uniform_(-np.sqrt(3.0 / dim), np.sqrt(3.0 / dim))

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel() # elements number
    max_len = max_len or lengths.max() # max_len
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat((batch_size, 1))
            .lt(lengths.unsqueeze(1)))

def softmax(self,input, axis=1):
    """
    Softmax applied to axis=n

    Args:
        input: {Tensor,Variable} input on which softmax is to be applied
        axis : {int} axis on which softmax is to be applied

    Returns:
        softmaxed tensors
    """

    input_size = input.size()
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


if __name__ == '__main__':
    print(torch.__version__)
