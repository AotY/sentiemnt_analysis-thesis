#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#
export CUDA_VISIBLE_DEVICES=0

mkdir -p data/
mkdir -p log/
mkdir -p models/

python train.py \
    --data_path data/cleaned.txt \
    --vocab_path data/vocab.word2idx.dict \
    --data_dir data/ \
    --log log/ \
    --embedding_size 128 \
    --hidden_size 256 \
    --num_layers 2 \
    --t_num_layers 2 \
    --transformer_size 256 \
    --k_size 64 \
    --v_size 64 \
    --inner_hidden_size 128 \
    --dense_size 128 \
    --num_heads 6 \
    --bidirectional \
    --dropout 0.2 \
    --lr 0.001 \
    --max_grad_norm 5.0 \
    --min_len 5 \
    --max_len 65 \
    --batch_size 32 \
    --valid_split 0.08 \
    --test_split 5 \
    --epochs 30 \
    --start_epoch 1 \
    --lr_patience 3 \
    --es_patience 5 \
    --device cuda \
    --seed 23 \
    --save_mode all \
    --save_model models/ \
    --smoothing \
    --model_type rnn \
    --mode train \
    #--checkpoint ./models/accu_43.850.pth \

    # conv2d ? -> 1
    # kernel_sizes = [(4, 512), (3, 512), (3, 256), (2, 256), (4, 512)]
    # output_channels = [512, 256, 256, 512, 512]
    # strides = [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1)]
    # maxpool_kernel_size = (4, 1)


/
