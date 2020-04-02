import torch
import torch.nn as nn


def create_deconv(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding=1,
                  activation='relu',
                  batch_norm=True,
                  bias=False):
    op = nn.ConvTranspose2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias)

    if activation == 'lrelu':
        act = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU(True)
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        deconv_layer = nn.Sequential(op, bn, act)
    else:
        deconv_layer = nn.Sequential(op, act)
    return deconv_layer


def create_conv(in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=1,
                activation='lrelu',
                batch_norm=True,
                bias=False):
    op = nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size,
                   stride=stride,
                   padding=padding,
                   bias=bias)

    if activation == 'lrelu':
        act = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'relu':
        act = nn.ReLU(True)

    if batch_norm == True:
        bn = nn.BatchNorm2d(out_channels)
        conv_layer = nn.Sequential(op, bn, act)
    else:
        conv_layer = nn.Sequential(op, act)
    return conv_layer
