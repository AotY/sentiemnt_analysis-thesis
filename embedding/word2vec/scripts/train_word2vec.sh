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

TEXT_DATA=$DATA_DIR/merged_data.txt
# VECTOR_DATA=$DATA_DIR/merged_data.vec.bin
VECTOR_DATA=$DATA_DIR/merged_data.vec.txt

#pushd ${SRC_DIR} && make; popd

#if [ ! -e $VECTOR_DATA ]; then
    #echo -----------------------------------------------------------------------------------------------------
    #echo -- Training vectors...
    #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 1 -size 128 -window 5 -negative 5 -hs 0 -sample 1e-3 -threads 12 -binary 0

#fi

#echo -----------------------------------------------------------------------------------------------------
#echo -- distance...

$BIN_DIR/distance $VECTOR_DATA

