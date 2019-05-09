#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 deep <deep@deep>
#
# Distributed under terms of the MIT license.

"""

"""

import json
import pickle


# load file

j = {
    'wordsim-240': {
        'web': {
            '1': {
                'cbow': 51.2479110,
                'sg': 57.49231943
            },
            '2': {
                'cbow': 56.7081038,
                'sg': 58.53300191
            },
            '3': {
                'cbow': 55.6863415
            },
            '4': {
                'cbow': 59.3807427
            }
        },
        'news': {
            '1': {
                'cbow': 54.0924209,
                'sg': 57.2577023,
            },
            '2': {
                'cbow': 57.1686710,
                'sg': 58.4815969
            },
            '3': {
                'cbow': 58.7591471
            },
            '4': {
                'cbow': 60.3397991
            }
        }
    },
    'wordsim-297': {
        'web': {
            '1': {
                'cbow': 54.2532115,
                'sg': 56.1966074
            },
            '2': {
                'cbow': 56.4114618,
                'sg': 58.62279271
            },
            '3': {
                'cbow': 56.6570859
            },
            '4': {
                'cbow': 59.3117402
            }
        },
        'news': {
            '1': {
                'cbow': 54.5332391,
                'sg': 57.7092162
            },
            '2': {
                'cbow': 57.6479145,
                'sg': 59.3228074
            },
            '3': {
                'cbow': 57.3317350
            },
            '4': {
                'cbow': 61.6130671
            }
        }
    }
}

def rho(names, v1=None, v2=None):
    return j[names[0]][names[1]][names[2]][names[3]]

if __name__ == '__main__':
    pass
