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
    --data_path data/label.cleaned.txt \
    --vocab_path data/vocab.word2idx.dict \
    --data_dir data/ \
    --visualization_dir visualization/ \
    --log log/ \
    --model_type self_attention \
    --n_classes 3 \
    --rnn_type LSTM \
    --embedding_size 128 \
    --hidden_size 256 \
    --num_layers 2 \
    --t_num_layers 2 \
    --transformer_size 128 \
    --k_size 64 \
    --v_size 64 \
    --inner_hidden_size 256 \
    --dense_size 128 \
    --num_heads 6 \
    --bidirectional \
    --in_channels 1 \
    --out_channels 128 \
    --kernel_heights 3 4 5 \
    --stride 1 \
    --padding 0 \
    --dropout 0.3 \
    --lr 0.0005 \
    --max_grad_norm 2.0 \
    --min_len 3 \
    --max_len 65 \
    --batch_size 32 \
    --valid_split 0.2 \
    --test_split 7 \
    --epochs 15 \
    --start_epoch 1 \
    --lr_patience 3 \
    --es_patience 5 \
    --device cuda \
    --seed 23 \
    --save_mode all \
    --save_model models/ \
    --mode train \
    #--checkpoint ./models/accuracy_93.688.pth \
    #--text 医生医术很恶心，而且态度太差了
    #--use_pos \
    # --smoothing \
    # --classes_weight 0.066 0.033 1 \


/
