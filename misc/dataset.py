#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import os
import pickle
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

from misc.vocab import PAD_ID, EOS_ID
from misc.sampler.imbalanced_dataset_sampler import ImbalancedDatasetSampler


def load_data(config, vocab):
    print('load data...')

    datas_pkl_path = os.path.join(config.data_dir, 'datas.pkl')
    if not os.path.exists(datas_pkl_path):
        datas = list()
        label_dates = {}
        with open(config.data_path, 'r') as f:
            for line in tqdm(f):
                line = line.rstrip()
                # disease, doctor, date, label, text = line.split('\t')
                label, text = line.split('\t')

                tokens = text.split()
                tokens = [token.split()[0] for token in tokens if len(token.split()) > 0]
                if len(tokens) < config.min_len:
                    continue

                ids = vocab.words_to_id(tokens)

                label = int(label)
                if config.problem == 'classification':
                    label -= 1

                if label_dates.get(label) is None:
                    label_dates[label] = list()

                label_dates[label].append((ids, label))

        total_sample = 0
        for label in label_dates.keys():
            np.random.shuffle(label_dates[label])
            total_sample += len(label_dates[label])

        label_max_smaple = int(total_sample * config.max_label_ratio)

        for label in label_dates.keys():
            if len(label_dates[label]) > label_max_smaple:
                datas.extend(label_dates[label][:label_max_smaple])
            else:
                datas.extend(label_dates[label])

        np.random.shuffle(datas)
        pickle.dump(datas, open(datas_pkl_path, 'wb'))
        print('datas: ', len(datas))
    else:
        datas = pickle.load(open(datas_pkl_path, 'rb'))

    return datas


def build_dataloader(config, datas):
    valid_split = int(config.valid_split * len(datas))
    test_split = int(config.batch_size * config.test_split)

    valid_dataset = Dataset(datas[:valid_split])
    test_dataset = Dataset(datas[valid_split: valid_split + test_split])
    train_dataset = Dataset(datas[valid_split + test_split:])

    collate_fn = MyCollate(config)

    # data loader
    imbalanced_sampler = ImbalancedDatasetSampler(train_dataset)
    config.classes_weight = imbalanced_sampler.labels_weight
    config.classes_count = imbalanced_sampler.labels_count

    train_sampler = None
    if config.sampler:
        train_sampler = imbalanced_sampler

    train_data = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False if config.sampler else True,
        num_workers=4,
        collate_fn=collate_fn,
        sampler=train_sampler
    )

    valid_data = data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,

    )

    test_data = data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    return train_data, valid_data, test_data, config


class Dataset(data.Dataset):
    def __init__(self, datas):
        self._datas = datas

    def __len__(self):
        return len(self._datas)

    def __getitem__(self, idx):
        ids, label = self._datas[idx]
        return ids, label


class MyCollate:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch_datas):
        max_len = self.config.max_len

        ''' Pad the instance to the max seq length in batch '''
        # sort by ids length
        batch_datas.sort(key=lambda item: len(item[0]), reverse=True)

        inputs, lengths, labels = list(), list(), list()
        inputs_pos = list()

        for ids, label in batch_datas:
            ids = ids[-min(max_len-1, len(ids)):]
            lengths.append(len(ids) + 1)

            # pad
            ids = ids + [EOS_ID] + [PAD_ID] * (max_len - len(ids))
            inputs.append(ids)

            pos = [pos_i + 1 if w_i !=
                   PAD_ID else PAD_ID for pos_i, w_i in enumerate(ids)]
            inputs_pos.append(pos)

            labels.append(label)

        # to [max_len, batch_size]
        inputs = torch.tensor(inputs, dtype=torch.long)
        inputs = inputs.transpose(0, 1)

        lengths = torch.tensor(lengths, dtype=torch.long)

        inputs_pos = torch.tensor(inputs_pos, dtype=torch.long)
        inputs_pos = inputs_pos.transpose(0, 1)

        # [batch_size]
        labels = torch.tensor(labels, dtype=torch.long)

        return inputs, lengths, labels, inputs_pos
