#!/bin/bash
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
THREADS=12

FIELD=web

TRAIN_WORD=$DATA_DIR/merge.$FIELD.txt
TRAIN_PINYIN=$DATA_DIR/merge.$FIELD.pinyin.txt
VOCAB_PATH=$DATA_DIR/merge.$FIELD.vocab.txt
IDF_PATH=$DATA_DIR/merge.$FIELD.idf.txt

declare -a model_types=(3 4)
#declare -a model_types=(1 2 3 4)
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
			vector_path=$DATA_DIR/merge.$mt.$csm.$cea.$size.bin
			./bin/word2vec -train-word $TRAIN_WORD -train-pinyin $TRAIN_PINYIN -train-idf $IDF_PATH -output $vector_path -save-vocab $VOCAB_PATH -size $size -binary $BINARY -cbow $cbow -window $WINDOW -debug 2 -hs $hs -negative $NEGATIVE -threads $THREADS -min-count 5 -model-type $mt 
			sleep 2
		done
	done
done

# declare -a ceas=("hs" "negative") # computationally efficient approximation
# if [$cea == "hs"]; then
	# hs=1
# else
	# hs=0
# fi


# $BIN_DIR/distance $VECTOR_PATH
