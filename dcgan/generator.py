import torch
import torch.nn as nn
from .layers import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")


class Generator(nn.Module):

    def __init__(self, name, noise_shape, conditional=False):
        super(Generator, self).__init__()
        self._name = name
        self._noise_shape = noise_shape
        self._conditional = conditional

    def get_noise(self, batch_size):
        size = [batch_size] + self._noise_shape
        noise = torch.randn(size=size).to(device)
        if self.training:
            noise.requires_grad_(True)
        else:
            with torch.no_grad():
                noise = noise
        return noise

    def _forward(self):
        raise NotImplementedError

    def forward(self, input_or_batch_size):
        if self._conditional:
            input = input_or_batch_size
            batch_size = input.size(0)
            noise = self.get_noise(batch_size)
            generated = self._forward(input, noise)
        else:
            batch_size = input_or_batch_size
            noise = self.get_noise(batch_size)
            generated = self._forward(noise)
        if not self.training:
            generated = generated.detach()
        return generated


class Gen_v0(Generator):

    def __init__(self, noise_size, channels, scale):
        super(Gen_v0, self).__init__(name='gen_v0',
                                     noise_shape=[noise_size, 1, 1],
                                     conditional=False)
        layers = [
            Conv_2D(noise_size, 8 * scale, 4, 1, 0, 'relu', transpose=True),
            Conv_2D(8 * scale, 4 * scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(4 * scale, 2 * scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(2 * scale, scale, 4, 2, 1, 'relu', transpose=True),
            Conv_2D(scale, channels, 4, 2, 1, 'tanh', False, transpose=True)
        ]
        self._layers = nn.ModuleList(layers)

    def _forward(self, noise):
        x = noise
        for layer in self._layers:
            x = layer(x)
        return x
