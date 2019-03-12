#!/usr/bin/env bash
#
# build_vocab.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#



python misc/build_vocab.py \
    --vocab_freq_path ./data/vocab.freq.txt \
    --vocab_path ./data/vocab.word2idx.dict \
    --vocab_size 3e4 \
    --min_count 3 \

/
