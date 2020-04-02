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
        layers = [
            Conv_2D(channels, scale, 4, 2, 1, 'lrelu', batch_norm=False),
            Conv_2D(scale, 2 * scale, 4, 2, 1, 'lrelu'),
            Conv_2D(2 * scale, 4 * scale, 4, 2, 1, 'lrelu'),
            Conv_2D(4 * scale, 8 * scale, 4, 2, 1, 'lrelu'),
            Conv_2D(8 * scale, 1, 4, 1, 0, 'id', batch_norm=False)
        ]

        self._layers = nn.ModuleList(layers)

    def _forward(self, image):
        x = image
        for layer in self._layers:
            x = layer(x)
        x = x[:, 0, 0, 0]
        return x
