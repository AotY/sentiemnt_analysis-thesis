#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

import seaborn as sns
# sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Arial']})
# sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})

sns.set_context(context="talk")
sns.set(font_scale=1.5)
sns.set(font='SimHei')

def draw(data, x, y, ax, cbar_ax=None, cbar=False):
    sns.heatmap(
        data,
        xticklabels=x, 
        yticklabels=y, 
        square=True, 
        ax=ax,
        vmin=0.0, 
        vmax=1.0,
        cbar=cbar,
        cbar_ax = cbar_ax,
        cbar_kws={"shrink": .32},
        # linewidths=0.7,
        # fmt='',
    )
