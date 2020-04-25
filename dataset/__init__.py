import json
import torch
import torchvision
from torchvision import transforms
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_dataset(args):
    if args.conditional:
        data_dir = args.data_folder
        preprocessing = transforms.Compose([
            transforms.Resize(64),
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
        shapes = torch.utils.data.DataLoader(shapes,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=0)

        return (lines, shapes), shapes_sampler
    else:
        preprocessing = transforms.Compose([
            transforms.Resize(64),
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
                                                  'resize ******256x256'))
        return data
