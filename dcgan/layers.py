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


class MeanPool(nn.Module):

    def __init__(self):
        super(MeanPool, self).__init__()

    def forward(self, x):
        x = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] +
             x[:, :, 1::2, 1::2]) / 4
        return x


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        (batch_size, height, width, depth) = x.size()
        output_depth = int(depth / self.block_size_sq)
        output_width = int(width * self.block_size)
        output_height = int(height * self.block_size)
        t_1 = x.reshape(batch_size, height, width, self.block_size_sq,
                        output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [
            t_t.reshape(batch_size, height, output_width, output_depth)
            for t_t in spl
        ]
        x = torch.stack(stacks,
                        0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(
                            batch_size, output_height, output_width,
                            output_depth)
        x = x.permute(0, 3, 1, 2)
        return output
