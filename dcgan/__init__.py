import torch
from .generator import *
from .discriminator import *
from .trainer import Trainer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_discriminator(args):
    builder = eval('Disc_v{}'.format(args.disc_version))
    print('* Building Discriminator ...')
    d_net = builder(args.channels, args.disc_scale,
                    args.conditional).to(device)
    if args.disc_checkpoint is not None:
        d_net.load_state_dict(
            torch.load(args.disc_checkpoint, map_location=device))
        print('----> Loading checkpoint = {}'.format(args.disc_checkpoint))
    return d_net


def get_generator(args):
    builder = eval('Gen_v{}'.format(args.gen_version))
    print('* Building Generator ...')
    g_net = builder(args.noise_size, args.channels, args.gen_scale,
                    args.conditional).to(device)
    if args.gen_checkpoint is not None:
        print(args.gen_checkpoint)
        g_net.load_state_dict(
            torch.load(args.gen_checkpoint, map_location=device))
        print('----> Loading checkpoint = {}'.format(args.gen_checkpoint))
    return g_net


def get_trainer(data, gen, disc, train_args):
    return Trainer(data, gen, disc, train_args)
