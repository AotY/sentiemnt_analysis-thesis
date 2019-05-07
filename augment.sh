#!/usr/bin/env bash
#
# augment.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#


python misc/augment.py \
    --data_path ./data/ChnSentiCorp_htl_all.txt \
    --save_path ./data/ChnSentiCorp_htl_all_augment.txt \
    --syno_path ./data/merge_syno.txt \
    --cilin_path ./data/cilin_ex.txt \
    --augment_nums 1 \
    --augment_labels 0 \


