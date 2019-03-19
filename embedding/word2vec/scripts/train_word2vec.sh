#!/usr/bin/env bash
#
# train_word2vec.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#

DATA_DIR=./data
BIN_DIR=./bin
SRC_DIR=./src

WORD_DATA=$DATA_DIR/merged_data.txt
PINYIN_DATA=$DATA_DIR/merged_data.pinyin.txt
# VECTOR_PATH=$DATA_DIR/merged_data.vec.bin
VECTOR_PATH=$DATA_DIR/merged_data.vec.txt
VOCAB_PATH=$DATA_DIR/merged_data.vocab.txt
PINYIN_VOCAB_PATH=$DATA_DIR/merged_data.pinyin.vocab.txt

#pushd ${SRC_DIR} && make; popd

if [ ! -e $VECTOR_PATH ]; then
    echo -----------------------------------------------------------------------------------------------------
    echo -- Training vectors...
    time $BIN_DIR/word2vec -train-word $WORD_DATA -output $VECTOR_PATH -cbow 1 -size 128 -window 5 -negative 5 -hs 0 -sample 1e-3 -threads 12 -binary 0 -save-vocab $VOCAB_PATH -save-pinyin-vocab $PINYIN_VOCAB_PATH -model-type 1 -train-pinyin $PINYIN_DATA

fi

#echo -----------------------------------------------------------------------------------------------------
#echo -- distance...

# $BIN_DIR/distance $VECTOR_PATH

