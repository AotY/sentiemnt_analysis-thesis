#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
            Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer=None):
        "Apply residual connection to any sublayer with the same size."
        if sublayer is None:
            return x + self.dropout(self.norm(x))
        else:
            return x + self.dropout(sublayer(self.norm(x)))
