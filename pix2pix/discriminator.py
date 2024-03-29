import torch
import torch.nn as nn
from .layers import *


class Discriminator(nn.Module):

    def __init__(self, version, conditional=False):
        super(Discriminator, self).__init__()
        self._version = version
        self._conditional = conditional

    def _forward(self):
        raise NotImplementedError

    def forward(self, inputs, inputs_bis=None):
        if inputs_bis == None:
            return self._forward(inputs)
        else:
            return self._forward(inputs), self._forward(inputs_bis)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Disc_v0(Discriminator):

    def __init__(self, channels, scale, conditional=False):
        super(Disc_v0, self).__init__(version=0, conditional=conditional)
        self._scale = scale
        if conditional:
            channels += 3
        self._channels = channels
        self._conv_layers = nn.Sequential(*[
            Conv_2D(channels, scale, 4, 2, 1, 'lrelu', batch_norm=False),
            Conv_2D(scale, 2 * scale, 4, 2, 1, 'lrelu'),
            Conv_2D(2 * scale, 4 * scale, 4, 2, 1, 'lrelu'),
            Conv_2D(4 * scale, 8 * scale, 4, 2, 1, 'lrelu'),
            Conv_2D(8 * scale, 1, 4, 1, 0, 'id', batch_norm=False)
        ])
        embed = [nn.Linear(1, 256),
                 View((-1, 1, 16, 16)),
                 torch.nn.Upsample(scale_factor=8, mode='bilinear')]
        self._embed_c = nn.Sequential(*embed)
        embed = [nn.Linear(1, 256),
                 View((-1, 1, 16, 16)),
                 torch.nn.Upsample(scale_factor=8, mode='bilinear')]
        self._embed_h = nn.Sequential(*embed)
        self._fc = nn.Linear(25, 1)

    def _forward(self, image):

        if type(image) == tuple:
            im, c, h = image
            c = self._embed_c(c)
            h = self._embed_h(h)
            image = torch.cat([im, c, h], 1)
        x = image
        x = self._conv_layers(x)
        x = nn.Flatten()(x)
        x = self._fc(x)
        x = x[:, 0]
        return x


class Disc_v1(Discriminator):

    def __init__(self, channels, scale, conditional=False):
        super(Disc_v1, self).__init__(version=1, conditional=conditional)
        self._scale = scale
        if conditional:
            channels += 1
        self._channels = channels
        self._conv_layers = nn.Sequential(*[
            Conv_2D(channels, scale, 3, 1, 1, 'id', False, False, True),
            ResidualBlock(scale, 2 * scale, 3, 'down', scale),
            ResidualBlock(2 * scale, 4 * scale, 3, 'down', int(scale / 2)),
            ResidualBlock(4 * scale, 8 * scale, 3, 'down', int(scale / 4)),
            ResidualBlock(8 * scale, 8 * scale, 3, 'down', int(scale / 8)),
        ])
        self._fc = nn.Linear(scale * scale * 2, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self._fc.weight)
        nn.init.constant_(self._fc.bias, 0.0)

    def _forward(self, image):
        x = image
        x = self._conv_layers(x)
        x = x.view(-1, self._scale * self._scale * 2)
        x = self._fc(x)
        x = x[:, 0]
        return x
