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

FIELD=news

TRAIN_WORD=$DATA_DIR/merge.$FIELD.txt
TRAIN_PINYIN=$DATA_DIR/merge.$FIELD.pinyin.txt
VOCAB_PATH=$DATA_DIR/merge.$FIELD.vocab.txt
PINYIN_VOCAB_PATH=$DATA_DIR/merge.$FIELD.pinyin.vocab.txt
IDF_PATH=$DATA_DIR/merge.$FIELD.idf.txt
TF_PATH=$DATA_DIR/merge.$FIELD.tf.txt

# declare -a model_types=(4 3 2 1)
# declare -a sizes=(100 200 300)
declare -a model_types=(4)
declare -a sizes=(200 300)
declare -a csms=("cbow" "sg") # continuous skip-gram models, (cbow, sg)

mt=1
cbow=0
size=100
cea=negative
hs=0
f=1

for mt in "${model_types[@]}"
do
	# or do whatever with individual element of the array
	for size in "${sizes[@]}"
	do
		for csm in "${csms[@]}"
		do
			if [ "$csm" == "sg" ]; then
				cbow=0
                if [ "$mt" -gt 2 ]; then
                    continue
                fi
			else
				cbow=1
			fi

			if [ "$FIELD" == "web" ]; then
				f=1
			else
				f=2
			fi

			vector_path=$DATA_DIR/merge.$FIELD.$mt.$csm.$cea.$size.bin
            # echo $vector_path

            ./bin/word2vec -train-word $TRAIN_WORD -train-pinyin $TRAIN_PINYIN -train-idf $IDF_PATH -train-tf $TF_PATH -output $vector_path -save-vocab $VOCAB_PATH -save-pinyin-vocab $PINYIN_VOCAB_PATH -size $size -binary $BINARY -cbow $cbow -window $WINDOW -debug 2 -hs $hs -negative $NEGATIVE -threads $THREADS -min-count 5 -model-type $mt -field-type $f
			sleep 1

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
