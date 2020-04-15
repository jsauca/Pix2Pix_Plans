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
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # cpu
# test
args = options.get_test_args()
# Generator
gen = dcgan.get_generator(args)

dir = os.path.join(args.outputs,
                   datetime.now().strftime('%m-%d_%H-%M-%S') + '/')
print('----> Creating directory = {}'.format(dir))
os.makedirs(dir)

RTV = RasterToVector()
RTV.load_state_dict(
    torch.load('rtv/checkpoints/rtv.pth', map_location=device))


def test(samples, RTV, rtv_=True):
    print('--> Generating {} samples ...'.format(dir))
    samples = gen(args.number)
    print('--> Saving samples = {}'.format(dir))
    for sample_idx, sample in enumerate(samples):
        vutils.save_image(
            sample,
            dir + '/' + 'sample_{}.png'.format(sample_idx))
    if rtv_:
        folder_inputs = dir
        folder_outputs = dir + '/rtv/'
        os.makedirs(folder_outputs)
        paths = [f for f in listdir(folder_inputs)
                 if isfile(join(folder_inputs, f)) and f.endswith('png')]
        full_rtv(folder_inputs, folder_outputs, paths, RTV)


test(samples, RTV)
