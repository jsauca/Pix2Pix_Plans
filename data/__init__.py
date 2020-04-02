import json
import torch
import torchvision
from torchvision import transforms
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_dataset(args):

    preprocessing = [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]
    if args.data_normalize:
        preprocessing.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    preprocessing = transforms.Compose(preprocessing)
    data = torchvision.datasets.ImageFolder(args.data_folder,
                                            transform=preprocessing)
    data = torch.utils.data.DataLoader(data,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       drop_last=True)
    return data