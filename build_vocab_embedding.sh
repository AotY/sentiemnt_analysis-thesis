#!/usr/bin/env bash
#
# build_vocab_embedding.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#



python misc/build_vocab_embedding.py \
    --vocab_path data/vocab.word2idx.dict \
    --wordvec_file data/GoogleNews-vectors-negative300.bin \
	--type word2vec \
	--embedding_size 300 \
	--save_path data/word2vec.vocab.npy \

/
