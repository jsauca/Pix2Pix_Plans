import json
import torch
import torchvision
import cv2
from torchvision import transforms
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_dataset(args):
    if args.conditional:
        data_dir = args.data_folder
        preprocessing = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        lines = torchvision.datasets.ImageFolder(data_dir + '/lines/',
                                                 transform=preprocessing)

        lines = torch.utils.data.DataLoader(lines,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=0)

        preprocessing = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(64),
            transforms.ToTensor(),
        ])
        shapes = torchvision.datasets.ImageFolder(data_dir + '/shapes/',
                                                  transform=preprocessing)
        shapes_sampler = torch.utils.data.RandomSampler(
            shapes, replacement=True, num_samples=args.num_samples)
        data_c = torch.load('dataset/images/data_c.pt').unsqueeze(1).float()
        data_h = torch.load('dataset/images/data_h.pt').unsqueeze(1).float()
        energy = torch.utils.data.TensorDataset(data_c, data_h)
        energy = torch.utils.data.DataLoader(energy,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=0)
        print('data_h', next(iter(data_h))[0].shape)
        shapes = torch.utils.data.DataLoader(shapes,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=0)

        return (lines, shapes, energy), shapes_sampler
    else:
        preprocessing = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        data = torchvision.datasets.ImageFolder(args.data_folder,
                                                transform=preprocessing)
        data = torch.utils.data.DataLoader(data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=0)
        print('* Loading dataset ...')
        print('----> Number of bacthes = {}'.format(len(data)))
        print('----> Preprocess = {} + {}'.format('normalize',
                                                  'resize 64x64'))
        return data


def get_dataset_test(args):
    data_dir = args.shape_folder
    preprocessing = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    shapes = torchvision.datasets.ImageFolder(data_dir + '/shapes',
                                              transform=preprocessing)
    shapes = torch.utils.data.DataLoader(shapes,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=True,
                                         num_workers=0)

    return shapes


def get_dataset_test_nrj(args):
    gen_input = open(args.gen_input, 'r')
    gen_input = [line.split(',') for line in gen_input.readlines()]
    shapes = [[cv2.imread(input[0], cv2.IMREAD_GRAYSCALE)]
              for input in gen_input]
    data_c = torch.from_numpy(
        np.array([float(input[1]) for input in gen_input])).unsqueeze(1)
    data_h = torch.from_numpy(
        np.array([float(input[2]) for input in gen_input])).unsqueeze(1)
    shapes = torch.from_numpy(np.array(shapes))
    print(shapes.shape)
    shapes = torch.utils.data.TensorDataset(shapes)

    energy = torch.utils.data.TensorDataset(data_c, data_h)
    energy = torch.utils.data.DataLoader(energy,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=True,
                                         num_workers=0)
    print('data_h', next(iter(data_h))[0].shape)
    shapes = torch.utils.data.DataLoader(shapes,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=True,
                                         num_workers=0)
    print('shapes', next(iter(shapes))[0].shape)
    return shapes, energy
