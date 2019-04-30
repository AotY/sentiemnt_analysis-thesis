#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#

#conda activate torch

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

mkdir -p data/
mkdir -p log/
mkdir -p models/


# declare -a model_types=("rnn" "cnn" "bert_avg" "bert_max" "bert_weight" "bert_sample_exp" "bert_gumbel_avg" "bert_gumbel_sum" "bert_gumbel_tau")
# declare -a model_types=("bert_gumbel_avg" "bert_gumbel_sum" "bert_gumbel_tau")
# declare -a model_types=("bert_gumbel_tau")

# --model_type bert_gumbel \
# for mt in "${model_types[@]}"
# do
# --model_type $mt \

python train.py \
    --data_path ./data/dmsc_v2_ratings.txt \
    --data_dir ./data/ \
    --vocab_size 15000 \
    --min_count 1 \
    --max_label_ratio 1.0 \
    --visualization_dir visualization/ \
    --log log/ \
    --problem classification \
    --model_type rnn \
    --rnn_type GRU \
    --n_classes 3 \
    --tau 0.5 \
    --embedding_size 100 \
    --hidden_size 128 \
    --num_layers 1 \
    --bidirectional \
    --t_num_layers 1 \
    --k_size 32 \
    --v_size 32 \
    --dense_size 128 \
    --inner_hidden_size 128 \
    --regression_dense_size 128 \
    --num_heads 4 \
    --in_channels 1 \
    --out_channels 100 \
    --kernel_heights 3 4 2 \
    --stride 1 \
    --padding 0 \
    --dropout 0.7 \
    --lr 0.001 \
    --min_len 3 \
    --max_len 90 \
    --batch_size 128 \
    --valid_split 0.15 \
    --test_split 1 \
    --epochs 25 \
    --start_epoch 1 \
    --lr_patience 3 \
    --es_patience 7 \
    --device cuda \
    --seed 23 \
    --save_mode best \
    --save_model models/ \
    --mode train \
    --use_penalization \
    --penalization_coeff 1.0 \
    --max_grad_norm 5.0 \
    --use_pos \
    # --use_pretrained_embedding \
    # --pre_trained_wv data/dmsc_v2_ratings_merge.web.1.cbow.negative.100.npy \
    #--sampler \
    #--warmup_proportion 0.1 \
    #--gradient_accumulation_steps 1 \
    # --n_warmup_steps 4000 \ #--checkpoint ./models/classification.accuracy_93.595.pth \
    #--text 没看多久，看得简单，感觉很一般。 \
    #--text 医生医术很高超，还有态度很棒！
    #--text 太差了，态度差，排队慢。
    # --classes_weight 0.557 1.0 0.038 \
    #--classes_weight  0.0082 0.00151 0.000053 \
    # --smoothing \
    # --transformer_size 100 \
    # --vocab_path data/vocab.word2idx.dict \

# done
