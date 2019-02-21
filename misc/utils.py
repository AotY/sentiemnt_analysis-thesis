#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Utils
"""
import os
import sys



def save_distribution(dist_list, save_path):
    with open(os.path.join(save_path), 'w', encoding="utf-8") as f:
        for i, j in dist_list:
            f.write('%s\t%s\n' % (str(i), str(j)))

