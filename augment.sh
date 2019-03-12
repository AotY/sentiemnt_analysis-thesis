#!/usr/bin/env bash
#
# augment.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#


python misc/augment.py \
    --data_path ./data/label.cleaned.txt \
    --save_path ./data/label.augment.cleaned.txt \
    --syno_path ./data/merge_syno.txt \
    --cilin_path ./data/cilin_ex.txt \
    --augment_num 3 \
    --augment_labels 1 \


