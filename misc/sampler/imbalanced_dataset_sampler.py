#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
ImbalancedDatasetSampler
https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
"""
import torch
import torch.utils.data as data
import torchvision


class ImbalancedDatasetSampler(data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None, replacement=True):
        self.replacement = replacement
        
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        samples_weight = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.samples_weight = torch.DoubleTensor(samples_weight)

        # weight for each label
        # min_label_count = min(label_to_count.values())
        min_label_count = 1.0
        labels_weight = [min_label_count / label_to_count[label] for label in range(len(label_to_count))]
        self.labels_weight = torch.DoubleTensor(labels_weight)
        self.labels_count = [label_to_count[label] for label in range(len(label_to_count))]

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            # print('dataset[idx] ----> ', dataset[idx])
            return dataset[idx][1]
            #  raise NotImplementedError

    def __iter__(self):
        """
        If replacement is ``True``, samples are drawn with replacement.

        If not, they are drawn without replacement, which means that when a
        sample index is drawn for a row, it cannot be drawn again for that row.
        """
        # return (self.indices[i] for i in
                # torch.multinomial(self.samples_weight, self.num_samples, replacement=self.replacement))

        return iter(torch.multinomial(self.samples_weight, self.num_samples, self.replacement).tolist())


    def __len__(self):
        return self.num_samples


'''

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=args.batch_size,
    **kwargs
)

'''
