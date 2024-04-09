from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Class for downsampling blocks
class DownsampleBlock(nn.Module):
    # each block consists of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        return x,indices

# Class for upsampling blocks
class UpsampleBlock(nn.Module):
    # each block consists of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        self.layers_down = nn.ModuleList([
            DownsampleBlock(3, 32),
            DownsampleBlock(32, 64),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
        ])

        # Construct upsampling layers
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        self.layers_up = nn.ModuleList([
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
            UpsampleBlock(32, 32),
        ])

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.final_conv = nn.Conv2d(32, 11, kernel_size=1)

    def forward(self, x):
        indices = []
        sizes = []

        # Downsampling
        for block in self.layers_down:
            sizes.append(x.size())
            x, ind = block(x)
            indices.append(ind)

        # Upsampling
        for block, size in zip(self.layers_up, sizes[::-1]):
            x = block(x, indices.pop(), output_size=size)

        x = self.final_conv(x)

        return x


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
