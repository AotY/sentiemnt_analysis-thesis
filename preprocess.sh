#!/usr/bin/env bash

python misc/preprocess.py \
    --data_dir ./../guahao_spider/data/ \
    --label_cleaned_path ./data/label.cleaned.txt \
    --score_cleaned_path ./data/score.cleaned.txt \
    --vocab_freq_path ./data/vocab.freq.txt \
    --save_dir ./data \
    --min_len 4 \
    --max_len 155 \

    /
