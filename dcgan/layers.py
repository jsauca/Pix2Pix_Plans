import torch
import torch.nn as nn

activations = {
    'relu': nn.ReLU(True),
    'lrelu': nn.LeakyReLU(0.2, inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'id': nn.Identity()
}


class Conv_2D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=1,
                 activation='relu',
                 batch_norm=True,
                 transpose=False,
                 bias=False):
        super(Conv_2D, self).__init__()
        if not transpose:
            self._op = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=bias)
        else:
            self._op = nn.ConvTranspose2d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=bias)

        self._act = activations[activation]
        if batch_norm:
            self._bn = nn.BatchNorm2d(out_channels)
        else:
            self._bn = nn.Identity()

    def forward(self, x):
        x = self._op(x)
        x = self._bn(x)
        x = self._act(x)
        return x
