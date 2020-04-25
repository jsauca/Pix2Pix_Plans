import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import torch
import torchvision.utils as vutils
from skimage import io, transform

import dcgan
from rtv.network import RasterToVector
from rtv.ip import *
import options
from eval import full_rtv

device = torch.device("cpu")

args = options.get_test_args()
gen = dcgan.get_generator(args)

dir = os.path.join(args.outputs,
                   datetime.now().strftime('%m-%d_%H-%M-%S') + '/')
print('----> Creating directory = {}'.format(dir))
os.makedirs(dir)

RTV = RasterToVector()
RTV.load_state_dict(
    torch.load('rtv/checkpoints/rtv.pth', map_location=device))


def test(RTV, conditional=True):
    print('--> Generating {} samples ...'.format(dir))
    # generate outputs
    # if conditional give one or several shapes
    samples = []
    if conditional:
        for shape in os.listdir(args.shape_folder):
            samples += gen(shape)
    else:
        samples = gen(args.number)
    exit()
    print('--> Saving samples = {}'.format(dir))
    for sample_idx, sample in enumerate(samples):
        vutils.save_image(
            sample,
            dir + 'sample_{}.png'.format(sample_idx))
    folder_inputs = dir
    folder_outputs = dir + '/rtv/'
    os.makedirs(folder_outputs)
    paths = [f for f in listdir(folder_inputs)
             if isfile(join(folder_inputs, f)) and f.endswith('png')]
    full_rtv(folder_inputs, folder_outputs, paths, RTV)


test(RTV)
