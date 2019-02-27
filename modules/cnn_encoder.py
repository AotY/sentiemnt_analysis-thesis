#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN Classification
https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/CNN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding):
        """
            Arguments
            ---------
            batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
            n_classes : 2 = (pos, neg)
            in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, max_len, embedding_size)
            out_channels : Number of output channels after convolution operation performed on the input matrix
            kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
            keep_probab : Probability of retaining an activation node during dropout operation
            vocab_size : Size of the vocabulary containing unique words
            embedding_size : Embedding dimension of GloVe word embeddings
            weights : Pre-trained GloVe embedding which we will use to create our word_embedding look-up table
            --------
        """
        super(CNNEncoder, self).__init__()
        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_heights = config.kernel_heights
        self.stride = config.stride
        self.padding = config.padding

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (self.kernel_heights[0], self.embedding_size),
            self.stride,
            self.padding
        )
        self.conv2 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (self.kernel_heights[1], self.embedding_size),
            self.stride,
            self.padding
        )
        self.conv3 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (self.kernel_heights[2], self.embedding_size),
            self.stride,
            self.padding
        )

        self.dropout = nn.Dropout(config.dropout)

        self.linear_final = nn.Linear(
            len(self.kernel_heights) * self.out_channels, config.n_classes)

    def conv_block(self, inputs, conv_layer):
        # print('inputs shape: ', inputs.shape)
        # [batch_size, out_channels, dim, 1]
        conv_out = conv_layer(inputs)
        # print('conv_out shape: ', conv_out.shape)

        # [batch_size, out_channels, dim]
        activation = F.relu(conv_out.squeeze(3))
        # print('activation shape: ', activation.shape)

        # [batch_size, out_channels], kernel_sizes: the size of the window to
        # take a max over
        max_out = F.max_pool1d(activation, activation.size(2)).squeeze(2)
        # print('max_out shape: ', max_out.shape)

        return max_out

    def forward(self, inputs, batch_size=None):
        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple.
        We perform convolution operation on the embedding matrix.
        whose shape for each batch is (max_len, embedding_size) with kernel of varying height but constant width which is same as the embedding_size.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors.
        This output is then fully connected to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        inputs: inputs of shape = (batch_size, num_sequences)
        batch_size : default = None.
        Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        [batch_size, n_classes]

        """

        # [batch_size, max_len, embedding_size]
        embedded = self.embedding(inputs)
        #  embedded = self.dropout(embedded)

        # [batch_size, 1, max_len, embedding_size]
        embedded = embedded.unsqueeze(1)

        max_out1 = self.conv_block(embedded, self.conv1)

        max_out2 = self.conv_block(embedded, self.conv2)

        max_out3 = self.conv_block(embedded, self.conv3)

        # [batch_size, num_kernels * out_channels]
        outputs = torch.cat((max_out1, max_out2, max_out3), dim=1)
        # print('outputs shape: ', outputs.shape)

        outputs = self.dropout(outputs)

        # [batch_size, num_kernels * out_channels]
        outputs = self.linear_final(outputs)
        # print('outputs shape: ', outputs.shape)

        return outputs, None
