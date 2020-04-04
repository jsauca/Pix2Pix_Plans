import numpy as np
from .utils import *
import math
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

model_url = 'http://dl.yf.io/drn/drn_d_54-0e0534ff.pth'


def upsample(input, size, mode):
    if True:
        return nn.functional.interpolate(input,
                                         size=size,
                                         mode=mode,
                                         align_corners=True)
    else:
        return nn.functional.upsample(input, size=size, mode=mode)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=False,
                     dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=(1, 1),
                 residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes,
                             planes,
                             stride,
                             padding=dilation[0],
                             dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,
                             planes,
                             padding=dilation[1],
                             dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=(1, 1),
                 residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation[1],
                               bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=-1,
                 out_middle=False,
                 pool_size=28,
                 arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3,
                                   channels[0],
                                   kernel_size=7,
                                   stride=1,
                                   padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock,
                                           channels[0],
                                           layers[0],
                                           stride=1)
            self.layer2 = self._make_layer(BasicBlock,
                                           channels[1],
                                           layers[1],
                                           stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3,
                          channels[0],
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False), nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True))

            self.layer1 = self._make_conv_layers(channels[0],
                                                 layers[0],
                                                 stride=1)
            self.layer2 = self._make_conv_layers(channels[1],
                                                 layers[1],
                                                 stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block,
                                       channels[4],
                                       layers[4],
                                       dilation=2,
                                       new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.pred = nn.Conv2d(self.out_dim,
                                  num_classes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.out_map < 32:
            self.out_pool = nn.MaxPool2d(32 // self.out_map)
            pass

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    new_level=True,
                    residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  dilation=(1, 1) if dilation == 1 else
                  (dilation // 2 if new_level else dilation, dilation),
                  residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      residual=residual,
                      dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes,
                          channels,
                          kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation,
                          bias=False,
                          dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map > 0:
            if self.num_classes > 0:
                if self.out_map == x.shape[2]:
                    x = self.pred(x)
                elif self.out_map > x.shape[2]:
                    x = self.pred(x)
                    x = upsample(input=x,
                                 size=(self.out_map, self.out_map),
                                 mode='bilinear')
                else:
                    x = self.out_pool(x)
                    y.append(x)
                    x = self.pred(x)
                    pass
            else:
                if self.out_map > x.shape[3]:
                    x = upsample(input=x,
                                 size=(self.out_map, self.out_map),
                                 mode='bilinear')
                    pass
                pass
        else:
            x = self.avgpool(x)
            x = self.pred(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


def drn_d_54(pretrained=False, out_map=256, num_classes=20, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1],
                arch='D',
                out_map=out_map,
                num_classes=num_classes,
                **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_url,
                                             model_dir='rtv/checkpoints/')
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'fc' not in k
        }
        state = model.state_dict()
        state.update(pretrained_dict)
        model.load_state_dict(state)
    return model


## Conv + bn + relu
class ConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 mode='conv',
                 use_bn=True):
        super(ConvBlock, self).__init__()

        self.use_bn = use_bn

        if padding == None:
            padding = (kernel_size - 1) // 2
            pass
        if mode == 'conv':
            self.conv = nn.Conv2d(in_planes,
                                  out_planes,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=False)
        elif mode == 'deconv':
            self.conv = nn.ConvTranspose2d(in_planes,
                                           out_planes,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        elif mode == 'conv_3d':
            self.conv = nn.Conv3d(in_planes,
                                  out_planes,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=False)
        elif mode == 'deconv_3d':
            self.conv = nn.ConvTranspose3d(in_planes,
                                           out_planes,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if self.use_bn:
            if '3d' not in mode:
                self.bn = nn.BatchNorm2d(out_planes)
            else:
                self.bn = nn.BatchNorm3d(out_planes)
                pass
            pass
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, inp):
        #return self.relu(self.conv(inp))
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))


## The pyramid module from pyramid scene parsing
class PyramidModule(nn.Module):

    def __init__(self,
                 in_planes,
                 middle_planes,
                 scales=[32, 16, 8, 4],
                 height=256,
                 width=256):
        super(PyramidModule, self).__init__()

        self.pool_1 = torch.nn.AvgPool2d(
            (scales[0] * height // width, scales[0]))
        self.pool_2 = torch.nn.AvgPool2d(
            (scales[1] * height // width, scales[1]))
        self.pool_3 = torch.nn.AvgPool2d(
            (scales[2] * height // width, scales[2]))
        self.pool_4 = torch.nn.AvgPool2d(
            (scales[3] * height // width, scales[3]))
        self.conv_1 = ConvBlock(in_planes,
                                middle_planes,
                                kernel_size=1,
                                use_bn=False)
        self.conv_2 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_3 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_4 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(scales[0] * height // width,
                                                scales[0]),
                                          mode='bilinear',
                                          align_corners=True)
        return

    def forward(self, inp):
        x_1 = self.upsample(self.conv_1(self.pool_1(inp)))
        x_2 = self.upsample(self.conv_2(self.pool_2(inp)))
        x_3 = self.upsample(self.conv_3(self.pool_3(inp)))
        x_4 = self.upsample(self.conv_4(self.pool_4(inp)))
        out = torch.cat([inp, x_1, x_2, x_3, x_4], dim=1)
        return out


## The module to compute plane depths from plane parameters
def calcPlaneDepthsModule(width, height, planes, metadata, return_ranges=False):
    urange = (torch.arange(width, dtype=torch.float32).cuda().view(
        (1, -1)).repeat(height, 1) / (float(width) + 1) *
              (metadata[4] + 1) - metadata[2]) / metadata[0]
    vrange = (torch.arange(height, dtype=torch.float32).cuda().view(
        (-1, 1)).repeat(1, width) / (float(height) + 1) *
              (metadata[5] + 1) - metadata[3]) / metadata[1]
    ranges = torch.stack(
        [urange, torch.ones(urange.shape).cuda(), -vrange], dim=-1)

    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)

    normalXYZ = torch.sum(ranges.unsqueeze(-2) *
                          planeNormals.unsqueeze(-3).unsqueeze(-3),
                          dim=-1)
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1).unsqueeze(-2).unsqueeze(
        -2) / normalXYZ
    planeDepths = torch.clamp(planeDepths, min=0, max=MAX_DEPTH)
    if return_ranges:
        return planeDepths, ranges
    return planeDepths


## The module to compute depth from plane information
def calcDepthModule(width, height, planes, segmentation, non_plane_depth,
                    metadata):
    planeDepths = calcPlaneDepthsModule(width, height, planes, metadata)
    allDepths = torch.cat(
        [planeDepths.transpose(-1, -2).transpose(-2, -3), non_plane_depth],
        dim=1)
    return torch.sum(allDepths * segmentation, dim=1)


## Compute matching with the auction-based approximation algorithm
def assignmentModule(W):
    O = calcAssignment(W.detach().cpu().numpy())
    return torch.from_numpy(O).cuda()


def calcAssignment(W):
    numOwners = int(W.shape[0])
    numGoods = int(W.shape[1])
    P = np.zeros(numGoods)
    O = np.full(shape=(numGoods,), fill_value=-1)
    delta = 1.0 / (numGoods + 1)
    queue = list(range(numOwners))
    while len(queue) > 0:
        ownerIndex = queue[0]
        queue = queue[1:]
        weights = W[ownerIndex]
        goodIndex = (weights - P).argmax()
        if weights[goodIndex] >= P[goodIndex]:
            if O[goodIndex] >= 0:
                queue.append(O[goodIndex])
                pass
            O[goodIndex] = ownerIndex
            P[goodIndex] += delta
            pass
        continue
    return O
