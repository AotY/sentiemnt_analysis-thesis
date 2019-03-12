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
    --syno_path ./data/syno.txt \
    --augment_num 2 \
    --augment_labels 1 \


