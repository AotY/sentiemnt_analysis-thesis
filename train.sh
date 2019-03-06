#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

mkdir -p data/
mkdir -p log/
mkdir -p models/

python train.py \
    --data_path data/label.cleaned.txt \
    --vocab_path data/vocab.word2idx.dict \
    --data_dir data/ \
    --visualization_dir visualization/ \
    --log log/ \
    --problem classification \
    --model_type self_attention \
    --n_classes 3 \
    --rnn_type LSTM \
    --embedding_size 256 \
    --hidden_size 256 \
    --num_layers 2 \
    --t_num_layers 1 \
    --transformer_size 256 \
    --k_size 64 \
    --v_size 64 \
    --inner_hidden_size 512 \
    --dense_size 128 \
    --regression_dense_size 128 \
    --num_heads 6 \
    --bidirectional \
    --in_channels 1 \
    --out_channels 256 \
    --kernel_heights 3 4 2 \
    --stride 1 \
    --padding 0 \
    --dropout 0.4 \
    --lr 0.001 \
    --min_len 1 \
    --max_len 55 \
    --batch_size 64 \
    --valid_split 0.2 \
    --test_split 2 \
    --epochs 30 \
    --start_epoch 1 \
    --lr_patience 5 \
    --es_patience 11 \
    --device cuda \
    --seed 23 \
    --save_mode all \
    --save_model models/ \
    --mode train \
    --use_pos \
    --use_penalization \
    --penalization_coeff 1.0 \
    # --max_grad_norm 5.0 \
    # --n_warmup_steps 4000 \
    #--checkpoint ./models/classification.accuracy_93.595.pth \
    #--text 没看多久，看得简单，感觉很一般。 \
    #--text 医生医术很高超，还有态度很棒！
    #--text 太差了，态度差，排队慢。
    #--sampler \
    # --classes_weight 0.557 1.0 0.038 \
    #--classes_weight  0.0082 0.00151 0.000053 \
    # --smoothing \


/
