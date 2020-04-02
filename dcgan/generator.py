import torch
import torch.nn as nn
from .layers import *


class Generator(nn.Module):

    def __init__(self, name, noise_shape, conditional=False):
        super(Generator, self).__init__()
        self._name = name
        self._noise_shape = noise_shape
        self._conditional = conditional

    def get_noise(self, batch_size, use_cuda=False):
        size = [batch_size] + self._noise_shape
        noise = torch.randn(size=size)
        if use_cuda:
            noise = noise.cuda()
        if self.training:
            noise.requires_grad_(True)
        else:
            with torch.no_grad():
                noise = noise
        return noise

    def _forward(self):
        raise NotImplementedError

    def forward(self, input_or_batch_size, use_cuda=False):
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
        self._deconv1 = create_deconv(noise_size, 8 * scale, 4, 1, 0)
        self._deconv2 = create_deconv(8 * scale, 4 * scale, 4, 2, 1)
        self._deconv3 = create_deconv(4 * scale, 2 * scale, 4, 2, 1)
        self._deconv4 = create_deconv(2 * scale, scale, 4, 2, 1)
        self._deconv5 = create_deconv(scale, channels, 4, 2, 1, 'tanh', False)

    def _forward(self, noise):
        z = noise
        x = self._deconv1(z)
        x = self._deconv2(x)
        x = self._deconv3(x)
        x = self._deconv4(x)
        x = self._deconv5(x)
        return x
