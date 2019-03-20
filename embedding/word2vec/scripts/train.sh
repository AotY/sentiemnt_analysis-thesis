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

NEGATIVE=5
WINDOW=5
BINARY=1

TRAIN_WORD=$DATA_DIR/merged_data.txt
TRAIN_PINYIN=$DATA_DIR/merged_data.pinyin.txt
VECTOR_PATH=$DATA_DIR/merged_data.vec.$MODEL_TYPE.$CSM_$CEA_$SIZE.bin
VOCAB_PATH=$DATA_DIR/merged_data.vocab.txt

declare -a model_types=(1 2)
declare -a sizes=(100 200 300)
declare -a csms=("cbow" "sg") # continuous skip-gram models, (cbow, sg)

mt=1
cbow=0
size=100
cea=negative
hs=0

for mt in "${model_types[@]}"
do
	# or do whatever with individual element of the array
	for size in "${sizes[@]}"
	do
		for csm in "${csms[@]}"
		do
			if [ "$csm" == "sg" ]; then
				cbow=0
			else
				cbow=1
			fi
			# ./bin/word2vec
			vector_path=$DATA_DIR/merged_data.vec.$mt.$csm.$cea.$size.bin
			./bin/word2vec -train-word $TRAIN_WORD -train-pinyin $TRAIN_PINYIN -output $vector_path -save-vocab $VOCAB_PATH -size $size -binary $BINARY -cbow $cbow -window $WINDOW -debug 2 -hs $hs -negative $NEGATIVE -threads 12 -min-count 5 -model-type $mt 
			sleep 2
		done
	done
done


# ./bin/word2vec -train-word $TRAIN_WORD -train-pinyin $TRAIN_PINYIN -output $VECTOR_PATH -save-vocab $VOCAB_PATH -size $SIZE -binary $BINARY -cbow $CBOW -window $WINDOW -debug 2 -negative $NEGATIVE -threads 12 -min-count 5 -model-type $MODEL_TYPE

# declare -a ceas=("hs" "negative") # computationally efficient approximation
# if [$cea == "hs"]; then
	# hs=1
# else
	# hs=0
# fi


# $BIN_DIR/distance $VECTOR_PATH
