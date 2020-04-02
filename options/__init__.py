import argparse
from .parse import *
import json
import os


def load_args(path):
    with open(path, 'r') as f:
        args = json.load(f)
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
    return args
