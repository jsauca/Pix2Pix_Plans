import torch
import torchvision
from torchvision import transforms
from .generator import *
from .discriminator import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_dataset(args):
    preprocessing = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data = torchvision.datasets.ImageFolder(args.data_folder,
                                            transform=preprocessing)
    data = torch.utils.data.DataLoader(data,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       drop_last=True)
    return data


def get_discriminator(args):
    builder = eval('Disc_v{}'.format(args.disc_version))
    d_net = builder(args.channels, args.disc_scale).to(device)
    if args.disc_checkpoint is not None:
        d_net.load_state_dict(
            torch.load(args.disc_checkpoint, map_location=device))
    return d_net


def get_generator(args):
    builder = eval('Gen_v{}'.format(args.gen_version))
    g_net = builder(args.noise_size, args.channels, args.gen_scale).to(device)
    if args.gen_checkpoint is not None:
        g_net.load_state_dict(
            torch.load(args.gen_checkpoint, map_location=device))
    return g_net
