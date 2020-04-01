import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim


def create_conv(in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=1,
                activation='lrelu',
                batch_norm=True):
    op = nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size,
                   stride=stride,
                   padding=padding)

    if activation == 'lrelu':
        act = nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'sigmoid':
        act = nn.Sigmoid()

    if batch_norm == True:
        bn = nn.BatchNorm2d(out_channels)
        conv_layer = nn.Sequential(op, bn, act)
    else:
        conv_layer = nn.Sequential(op, act)
    return conv_layer


class Discriminator(nn.Module):

    def __init__(self, name, conditional=False):
        self._name = name
        self._input_shape = input_shape
        self._conditional = conditional

    def _forward(self):
        raise NotImplementedError

    def forward(self, *inputs):
        if len(inputs) > 1:
            return list(map(self._forward, inputs))
        else:
            return self._forward(inputs[0])


class Disc_v0(nn.Module):

    def __init__(self):
        super(Disc_v0, self).__init__(name='disc_v0')
        self._conv1 = create_conv(3, 128, 4, 2, batch_norm=False)
        self._conv2 = create_conv(128, 256, 4, 2)
        self._conv3 = create_conv(256, 512, 4, 2)
        self._conv4 = create_conv(512, 1024, 4, 2)
        self._conv5 = create_conv(1024, 1, 4, 1, padding=0, batch_norm=False)
        self._fc = nn.Sequential(*[nn.Flatten(), nn.Linear(13 * 13, 1)])

    def _forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = self._fc(x)
        return x
