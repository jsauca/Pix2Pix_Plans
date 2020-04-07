from torch import nn
from .layers import *


class RasterToVector(nn.Module):

    def __init__(self, height=256, width=256):
        super(RasterToVector, self).__init__()
        self.drn = drn_d_54(pretrained=True,
                            out_map=32,
                            num_classes=-1,
                            out_middle=False)
        self.height = height
        self.width = width
        self.pyramid = PyramidModule(512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = nn.Conv2d(512,
                                           NUM_CORNERS + NUM_ICONS + 2 +
                                           NUM_ROOMS + 2,
                                           kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(height, width), mode='bilinear')

    def forward(self, inp):
        features = self.drn(inp)
        features = self.pyramid(features)
        features = self.feature_conv(features)
        segmentation = self.upsample(self.segmentation_pred(features))
        segmentation = segmentation.transpose(1, 2).transpose(2, 3).contiguous()
        return torch.sigmoid(
            segmentation[:, :, :, :NUM_CORNERS]
        ), segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS +
                        2], segmentation[:, :, :, -(NUM_ROOMS + 2):]
