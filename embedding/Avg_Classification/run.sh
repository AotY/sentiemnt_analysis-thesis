#!/usr/bin/env bash
#
# run.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#


python Avg_Classification/avg_classification.py \
    --data_path ../data/ChnSentiCorp_htl_all.txt \
    --embedding_path ./embedding/word2vec/data/merge.web.2.sg.negative.100.bin \
    --test_size 0.2 \
	--min_len 3 \

/


