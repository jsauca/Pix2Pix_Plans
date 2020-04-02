import torch
import torch.nn as nn
from .layers import *


class Discriminator(nn.Module):

    def __init__(self, name, conditional=False):
        super(Discriminator, self).__init__()
        self._name = name
        self._conditional = conditional

    def _forward(self):
        raise NotImplementedError

    def forward(self, inputs, inputs_bis=None):
        if inputs_bis == None:
            return self._forward(inputs)
        else:
            return self._forward(inputs), self._forward(inputs_bis)


class Disc_v0(Discriminator):

    def __init__(self, channels, scale):
        super(Disc_v0, self).__init__(name='disc_v0')
        self._conv1 = create_conv(channels, scale, 4, 2, 1, batch_norm=False)
        self._conv2 = create_conv(scale, 2 * scale, 4, 2, 1)
        self._conv3 = create_conv(2 * scale, 4 * scale, 4, 2, 1)
        self._conv4 = create_conv(4 * scale, 8 * scale, 4, 2, 1)
        self._conv5 = create_conv(8 * scale, 1, 4, 1, 0, batch_norm=False)

    def _forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)[:, 0, 0, 0]
        return x
