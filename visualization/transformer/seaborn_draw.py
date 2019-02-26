#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


def draw(data, x, y, ax):
    seaborn.heatmap(
        data,
        xticklabels=x, 
        yticklabels=y, 
        ax=ax
        square=True, 
        vmin=0.0, 
        vmax=1.0,
        cbar=False,
    )
