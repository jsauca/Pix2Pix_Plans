import torch
import torch.nn as nn
from .layers import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Generator(nn.Module):

    def __init__(self, version, noise_shape, conditional=False):
        super(Generator, self).__init__()
        self._version = version
        self._noise_shape = noise_shape
        self._conditional = conditional

    def get_noise(self, batch_size, training):
        size = [batch_size] + self._noise_shape
        noise = torch.randn(size=size).to(device)
        if training:
            noise.requires_grad_(True)
        else:
            with torch.no_grad():
                noise = noise
        return noise

    def _forward(self):
        raise NotImplementedError

    def forward(self, input_or_batch_size, training=False):
        if self._conditional:
            input = input_or_batch_size
            batch_size = input.size(0)
            noise = self.get_noise(batch_size, training)
            generated = self._forward(input, noise)
        else:
            batch_size = input_or_batch_size
            noise = self.get_noise(batch_size, training)
            generated = self._forward(noise)
        if not self.training:
            generated = generated.detach()
        return generated


class Gen_v0(Generator):

    def __init__(self, noise_size, channels, scale):
        super(Gen_v0, self).__init__(version=0,
                                     noise_shape=[noise_size, 1, 1],
                                     conditional=False)
        self._scale = scale
        self._channels = channels
        self._deconv_layers = nn.Sequential(*[
            Conv_2D(noise_size, 8 * scale, 4, 1, 0, 'relu', transpose=True),
            Conv_2D(8 * scale, 4 * scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(4 * scale, 2 * scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(2 * scale, scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(scale, channels, 4, 2, 1, 'tanh', False, transpose=True)
        ])

    def _forward(self, z):
        x = self._deconv_layers(z)
        return x


class Gen_v1(Generator):

    def __init__(self, noise_size, channels, scale):
        super(Gen_v1, self).__init__(version=1,
                                     noise_shape=[noise_size],
                                     conditional=False)
        self._scale = scale
        self._channels = channels
        self._fc = nn.Linear(noise_size, scale * scale * 2)
        self._deconv_layers = nn.Sequential(*[
            ResidualBlock(8 * scale, 8 * scale, 3, 'up', scale),
            ResidualBlock(8 * scale, 4 * scale, 3, 'up', scale),
            ResidualBlock(4 * scale, 2 * scale, 3, 'up', scale),
            ResidualBlock(2 * scale, scale, 3, 'up', scale),
            nn.BatchNorm2d(scale),
            nn.ReLU(),
            Conv_2D(scale, channels, 3, 1, 1, 'tanh', False)
        ])

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self._fc.weight)
        nn.init.constant_(self._fc.bias, 0.0)

    def _forward(self, z):
        x = self._fc(z)
        x = x.view(-1, 8 * self._scale, int(self._scale / 16),
                   int(self._scale / 16))
        x = self._deconv_layers(x)
        return x
