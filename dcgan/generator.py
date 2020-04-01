import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim


def create_deconv(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding=1,
                  activation='lrelu',
                  batch_norm=True):
    op = nn.ConvTranspose2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding)

    if activation == 'lrelu':
        act = nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        deconv_layer = nn.Sequential(op, bn, act)
    else:
        deconv_layer = nn.Sequential(op, act)
    return deconv_layer


class Generator(nn.Module):

    def __init__(self, name, noise_shape, conditional=False):
        self._name = name
        self._noise_shape = noise_shape
        self._conditional = conditional

    def get_noise(self, batch_size, use_cuda=False):
        size = [batch.size] + self._noise_shape
        noise = torch.randn(size=size)
        if use_cuda:
            noise = noise.cuda()
        return noise

    def _forward(self):
        raise NotImplementedError

    def forward(self, input_or_batch_size, use_cuda=False):
        if self._conditional:
            input = input_or_batch_size
            batch_size = input.size(0)
            noise = self.get_noise(batch_size)
            return self._forward(input, noise)
        else:
            batch_size = input_or_batch_size
            noise = self.get_noise(batch_size)
            return self._forward(noise)


class Gen_v0(Generator):

    def __init__(self, noise_size):
        super(Gen_v0, self).__init__(name='gen_v0',
                                     noise_shape=[self._noise_size, 13, 13],
                                     conditional=False)
        self._deconv1 = create_deconv(self._noise_size, 1024, 4, 1, padding=0)
        self._deconv2 = create_deconv(1024, 512, 4, 2)
        self._deconv3 = create_deconv(512, 256, 4, 2)
        self._deconv4 = create_deconv(256, 128, 4, 2)
        self._deconv5 = create_deconv(128, 3, 4, 2, activation='tanh')

    def _forward(self, noise):
        z = noise
        x = self._deconv1(z)
        x = self._deconv2(x)
        x = self._deconv3(x)
        x = self._deconv4(x)
        x = self._deconv5(x)
        return x
