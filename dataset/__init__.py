import json
import torch
import torchvision
from torchvision import transforms
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_dataset(args):
    if args.conditional:
        preprocessing = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        lines = torchvision.datasets.ImageFolder('images/lines/',
                                                 transform=preprocessing)

        lines = torch.utils.data.DataLoader(lines,
                                            batch_size=64,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=0)

        preprocessing = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(64),
            transforms.ToTensor(),
        ])
        shapes = torchvision.datasets.ImageFolder('images/shapes/',
                                                  transform=preprocessing)

        shapes = torch.utils.data.DataLoader(shapes,
                                             batch_size=64,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=0)
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
        print('----> Preprocess = {} + {}'.format('normalize', 'resize 256x256'))
        return data
