# -*- coding: utf-8 -*-
"""DCGANPytorch.ipynb"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
from dataset import sizer
"""  # Hyperparameters"""

USE_GPU = False
NOISE_SIZE = 100
PREPROCESS = True
BATCH_SIZE = 128
OPTIMIZER = 'Adam'
LEARNING_RATE = 0.0002

"""## Utils"""


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, cmap='gray')
    return plt


class Data:
    def __init__(self,
                 preprocess=PREPROCESS):
        if preprocess:
            preprocessing = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])
        else:
            preprocessing = None
        self._dataset = torchvision.datasets.ImageFolder(
            'data/', transform=preprocessing)
        self._length = len(self._dataset)

    def sample(self, num_samples):
        idxs = np.random.randint(self._length, size=num_samples)
        samples = self._dataset[idxs].unsqueeze(1)
        return samples


"""## Networks"""


def create_deconv(in_features,
                  out_features,
                  kernel_size,
                  stride,
                  padding=1,
                  activation='lrelu',
                  batch_norm=True):
    op = nn.ConvTranspose2d(in_features,
                            out_features,
                            kernel_size,
                            stride=stride,
                            padding=padding)

    if activation == 'lrelu':
        act = nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'tanh':
        act = nn.Tanh()
    if batch_norm:
        bn = nn.BatchNorm2d(out_features)
        deconv_layer = nn.Sequential(op, bn, act)
    else:
        deconv_layer = nn.Sequential(op, act)
    return deconv_layer

# Generator


class Generator(nn.Module):
    def __init__(self, noise_size=NOISE_SIZE):
        super(Generator, self).__init__()
        self._noise_size = noise_size
        self._deconv1 = create_deconv(noise_size, 1024, 4, 1, padding=1)
        self._deconv2 = create_deconv(1024, 512, 4, 2)
        self._deconv3 = create_deconv(512, 256, 4, 2)
        self._deconv4 = create_deconv(256, 128, 4, 2)
        self._deconv5 = create_deconv(128, 1, 4, 2, 3, activation='tanh')

    def forward(self, z):
        x = self._deconv1(z)
        x = self._deconv2(x)
        x = self._deconv3(x)
        x = self._deconv4(x)
        x = self._deconv5(x)
        return x


def create_conv(in_features,
                out_features,
                kernel_size,
                stride,
                padding=1,
                activation='lrelu', batch_norm=True):
    op = nn.Conv2d(in_features,
                   out_features,
                   kernel_size,
                   stride=stride,
                   padding=padding)

    if activation == 'lrelu':
        act = nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'sigmoid':
        act = nn.Sigmoid()

    if batch_norm == True:
        bn = nn.BatchNorm2d(out_features)
        conv_layer = nn.Sequential(op, bn, act)
    else:
        conv_layer = nn.Sequential(op, act)
    return conv_layer

# Discriminator


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._conv1 = create_conv(1, 128, 4, 2, batch_norm=False)
        self._conv2 = create_conv(128, 256, 4, 2)
        self._conv3 = create_conv(256, 512, 4, 2)
        self._conv4 = create_conv(512, 1024, 4, 2)
        self._conv5 = create_conv(
            1024, 1, 4, 2, padding=2, activation='sigmoid', batch_norm=False)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x).squeeze()
        return x


"""## Training and test"""


class Trainer:
    def __init__(self,
                 data,
                 generator,
                 discriminator,
                 batch_size=BATCH_SIZE,
                 optimizer=OPTIMIZER,
                 learning_rate=LEARNING_RATE,
                 noise_size=NOISE_SIZE):
        self._data = data
        self._batch_size = batch_size
        self._noise_size = noise_size
        self._g_net = generator.float()
        self._d_net = discriminator.float()
        self._data = data

        if optimizer == 'Adam':
            optimizer_class = optim.Adam
        elif optimizer == 'SGD':
            optimizer_class = optim.SGD
        self._g_opt = optimizer_class(
            self._g_net.parameters(), lr=learning_rate)
        self._d_opt = optimizer_class(
            self._d_net.parameters(), lr=learning_rate)

    def _g_step(self):

        noise = torch.randn(size=(self._batch_size, self._noise_size, 1, 1))
        labels = torch.ones((self._batch_size, 1), dtype=int)
        x_fake = self._g_net(noise)
        d_fake = self._d_net(x_fake).unsqueeze(1)
        self._g_opt.zero_grad()
        loss = nn.BCELoss()(d_fake.float(), labels.float())
        loss.backward()
        self._g_opt.step()

    def _d_step(self):
        noise = torch.randn(size=(self._batch_size // 2,
                                  self._noise_size, 1, 1)).float()
        labels_zeros = torch.zeros(
            (self._batch_size // 2, 1), dtype=int).float()
        labels_ones = torch.ones((self._batch_size // 2, 1), dtype=int).float()
        x_fake = self._g_net(noise)
        d_fake = self._d_net(x_fake.float())
        loss_fake = nn.BCELoss()(d_fake.unsqueeze(1), labels_zeros)
        x_real = self._data.sample(self._batch_size // 2).double()
        d_real = self._d_net(x_real.float())

        loss_real = nn.BCELoss()(d_real.unsqueeze(1), labels_ones)
        loss = loss_real + loss_fake
        self._d_opt.zero_grad()
        loss.backward()
        self._d_opt.step()

    def step(self):
        self._g_step()
        self._d_step()

    def test(self):
        noise = torch.randn(size=(1, self._noise_size, 1, 1))
        x_fake = self._g_net(noise).squeeze().data.numpy()
        gen_image(x_fake)


"""## Implementation"""

disc = Discriminator()
gen = Generator()
data = Data()
trainer = Trainer(data, gen, disc)

for it in range(1000):

    trainer.step()
    if it % 100 == 0:
        print('iteration_'.format(it))
        trainer.test()
