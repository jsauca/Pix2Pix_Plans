import argparse
from .parse import *
import json
import os
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def load_args(path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


def save_args(args, path):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def get_train_args():
    parser = argparse.ArgumentParser()
    options_data(parser)
    options_disc(parser)
    options_gen(parser)
    options_trainer(parser)
    args = parser.parse_args()
    print('* Loading options ...')
    for arg, value in args.__dict__.items():
        print('----> {} = {}'.format(arg, value))
    print('----> {} = {}'.format('use_cuda', use_cuda))
    print('----> {} = {}'.format('device', device))
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    options_test_nrj(parser)
    args = parser.parse_args()
    print('* Loading options ...')
    for arg, value in args.__dict__.items():
        print('----> {} = {}'.format(arg, value))
    return args
