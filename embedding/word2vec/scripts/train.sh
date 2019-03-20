#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#


DATA_DIR=./data
BIN_DIR=./bin
SRC_DIR=./src

MODEL_TYPE=1
CBOW=0
NEGATIVE=5
WINDOW=5
BINARY=1
SIZE=128

TRAIN_WORD=$DATA_DIR/merged_data.txt
TRAIN_PINYIN=$DATA_DIR/merged_data.pinyin.txt
VECTOR_PATH=$DATA_DIR/merged_data.vec.1_sg_negative_128.bin


./bin/word2vec -train-word ./data/merged_data.txt -train-pinyin $TRAIN_PINYIN -output $VECTOR_PATH -size $SIZE -binary $BINARY -cbow $CBOW -window $WINDOW -debug 2 -negative $NEGATIVE -threads 12 -min-count 5 -model-type $MODEL_TYPE


$BIN_DIR/distance $VECTOR_PATH
