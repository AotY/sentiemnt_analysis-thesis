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

j = {
	'web': {
			'1': {
				'cbow': 21.2989613,
				'sg': 26.0737610
			},
			'2': {
				'cbow': 16.6379102,
				'sg': 25.7811014
			},
			'3': {
				'cbow': 35.2209156
			},
			'4': {
				'cbow': 27.5946575
			}
		},
		'news': {
			'1': {
				'cbow': 45.9418308,
				'sg': 53.8098218
			},
			'2': {
				'cbow': 50.0270830,
				'sg': 52.3163023
			},
			'3': {
				'cbow': 57.6982014
			},
			'4': {
				'cbow': 58.0139021
			}
		}
}

def format(num, names=None):
    if names is None:
        return num
    else:
        return j[names[0]][names[1]][names[2]]
