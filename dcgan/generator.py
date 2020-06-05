import torch
import torch.nn as nn
from .layers import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Generator(nn.Module):

    def __init__(self, version, noise_size, conditional=False):
        super(Generator, self).__init__()
        self._version = version
        self._noise_size = noise_size
        self._conditional = conditional
        if conditional:
            self._build_cgan()

    def _build_cgan(self):
        self._cgan = nn.Sequential(nn.Conv2d(1, 1, 3, stride=2, padding=1),
                                   nn.Flatten(),
                                   nn.Linear(32 * 32, self._noise_size))

    def get_noise(self, batch_size, training):
        size = [batch_size, self._noise_size]
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
            input = input_or_batch_size[0]
            c = input_or_batch_size[1]
            h = input_or_batch_size[2]
            batch_size = input.size(0)
            noise = self.get_noise(batch_size, training)
            prefix = self._cgan(input)
            if not training:
                prefix = prefix.detach()

            gen_input = torch.cat([noise, prefix, c, h], 1)
        else:
            batch_size = input_or_batch_size
            gen_input = self.get_noise(batch_size, training)
        generated = self._forward(gen_input)
        if not self.training:
            generated = generated.detach()
        return generated


class Gen_v0(Generator):

    def __init__(self, noise_size, channels, scale, conditional):
        super(Gen_v0, self).__init__(version=0,
                                     noise_size=noise_size,
                                     conditional=conditional)
        self._scale = scale
        self._noise_size = noise_size
        self._channels = channels
        if conditional:
            in_channels = noise_size * 2 + 2
        else:
            in_channels = noise_size
        self._deconv_layers = nn.Sequential(*[
            Conv_2D(in_channels, 8 * scale, 4, 1, 0, 'relu', transpose=True),
            Conv_2D(8 * scale, 4 * scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(4 * scale, 2 * scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(2 * scale, scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(scale, channels, 4, 2, 1, 'tanh', False, transpose=True)
        ])

    def _forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        print(100 * '$$$$', z.size)
        x = self._deconv_layers(z)
        return x


class Gen_v1(Generator):

    def __init__(self, noise_size, channels, scale, conditional):
        super(Gen_v1, self).__init__(version=1,
                                     noise_size=noise_size,
                                     conditional=conditional)
        self._scale = scale
        self._channels = channels
        if conditional:
            in_features = noise_size * 2
        else:
            in_features = noise_size
        self._fc = nn.Linear(in_features, scale * scale * 2)
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
