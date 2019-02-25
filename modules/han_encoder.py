#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Hierarchical Attention Networks for Document Classification
https://github.com/cedias/Hierarchical-Sentiment/blob/master/Nets.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import rnn_factory, rnn_init


class EmbedAttention(nn.Module):
    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.att_w = nn.Linear(att_size, 1, bias=False)

    def forward(self, input, len_s):
        att = self.att_w(input).squeeze(-1)
        out = self._masked_softmax(att, len_s).unsqueeze(-1)
        return out

    def _masked_softmax(self, mat, len_s):

        # print(len_s.type())
        len_s = len_s.type_as(mat.data)  # .long()
        idxes = torch.arange(0, int(len_s[0]), out=mat.data.new(
            int(len_s[0])).long()).unsqueeze(1)
        mask = (idxes.float() < len_s.unsqueeze(0)).float()

        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(0, True) + 0.0001

        return exp/sum_exp.expand_as(exp)


class AttentionalBiRNN(nn.Module):

    def __init__(self,
                 config):
        #  inp_size, hid_size, dropout=0, RNN_cell=nn.GRU):
        super(AttentionalBiRNN, self).__init__()

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num
        self.n_classes = config.n_classes

        # rnn
        self.rnn = rnn_factory(
            rnn_type=config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )
        rnn_init(config.rnn_type, self.rnn)

        self.lin = nn.Linear(self.hidden_size, self.hidden_size)
        self.att_w = nn.Linear(self.hidden_size, 1, bias=False)
        self.emb_att = EmbedAttention(self.hidden_size)

    def forward(self, packed_batch):
        rnn_sents, _ = self.rnn(packed_batch)
        enc_sents, len_s = nn.utils.rnn.pad_packed_sequence(rnn_sents)

        emb_h = F.tanh(self.lin(enc_sents))

        attended = self.emb_att(emb_h, len_s) * enc_sents
        return attended.sum(0, True).squeeze(0)


class HANEncoder(nn.Module):

    def __init__(self,
                 config,
                 embedding):
                 #  ntoken, num_class, emb_size=200, hid_size=50):
        super(HANEncoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.hidden_size = config.hidden_size
        self.n_classes = config.n_classes

        self.word_attn = AttentionalBiRNN(self.embedding_size, self.hidden_size)
        self.sent_attn = AttentionalBiRNN(self.hidden_size, self.hidden_size)
        self.lin_out = nn.Linear(self.hidden_size, self.n_classes)

    def set_emb_tensor(self, emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

    def _reorder_sent(self, sents, sent_order):

        sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))

        return revs
