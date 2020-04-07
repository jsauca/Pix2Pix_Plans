import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
import numpy as np
import os
from datetime import datetime
import torch
from .utils import *
from .ip import *
from skimage import io, transform
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Evaluator:

    def __init__(self, args):
        self._args = args
        self._build_rtv()

    def _build_rtv(self):
        self._rtv = RasterToVector().eval()
        self._rtv.load_state_dict(
            torch.load('rtv/checkpoints/rtv.pth',
                       map_location=torch.device('cpu')))
        for param in self._rtv.parameters():
            param.requires_grad = False

    def _get_input_from_path(self, path):
        image = io.imread(path)
        if image.shape[2] == 4:
            image[np.where(image[:, :, 3] == 0)] = 255
        image = transform.resize(image, (256, 256))
        image = image[:, :, :3].astype('float32')
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image

    def _get_inputs_from_folder(self, folder):
        images = [
            self._load_input_from_path(self, path)
            for path in os.listdir(folder)
        ]
        return images

    def _get_inputs_from_tensor(self, tensor):
        if len(tensor.shape) == 4 and tensor.size(0) != 1:
            return [t.unsqueeze(0) for t in list(tensor)]
        elif len(tensor.shape) == 3:
            return tensor.unsqueeze(0)
        else:
            return tensor

    def reconstruct_plans(self, image, output_prefix):
        with torch.no_grad():
            corner_pred, icon_pred, room_pred = self._rtv(image)
            corner_pred, icon_pred, room_pred = corner_pred.squeeze(
                0), icon_pred.squeeze(0), room_pred.squeeze(0)
            corner_heatmaps = corner_pred.detach().cpu().numpy()
            icon_heatmaps = F.softmax(icon_pred, dim=-1).detach().cpu().numpy()
            room_heatmaps = F.softmax(room_pred, dim=-1).detach().cpu().numpy()
            reconstructFloorplan(corner_heatmaps[:, :, :13],
                                 corner_heatmaps[:, :, 13:17],
                                 corner_heatmaps[:, :, -4:],
                                 icon_heatmaps,
                                 room_heatmaps,
                                 output_prefix=output_prefix,
                                 densityImage=None,
                                 gt_dict=None,
                                 gt=False,
                                 gap=-1,
                                 distanceThreshold=-1,
                                 lengthThreshold=-1,
                                 debug_prefix='test',
                                 heatmapValueThresholdWall=None,
                                 heatmapValueThresholdDoor=None,
                                 heatmapValueThresholdIcon=None,
                                 enableAugmentation=True)

        dicts = {
            'corner': corner_pred.max(-1)[1].detach().cpu().numpy(),
            'icon': icon_pred.max(-1)[1].detach().cpu().numpy(),
            'room': room_pred.max(-1)[1].detach().cpu().numpy()
        }

        for info in ['corner', 'icon', 'room']:
            cv2.imwrite(
                output_prefix + '.png',
                drawSegmentationImage(dicts[info],
                                      blackIndex=0,
                                      blackThreshold=0.5))
