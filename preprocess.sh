#!/usr/bin/env bash

python misc/preprocess.py \
    --data_dir ./../guahao_spider/data/ \
    --cleaned_path ./data/cleaned.txt \
    --vocab_freq_path ./data/vocab.freq.txt \
    --save_dir ./data \
    --min_len 5 \
    --max_len 155 \

    /
