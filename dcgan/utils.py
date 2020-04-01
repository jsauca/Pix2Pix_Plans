import torch
import torchvision
from torchvision import transforms


def build_dataset(folder, batch_size):
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data = torchvision.datasets.ImageFolder(folder, transform=preprocessing)
    data = torch.utils.data.DataLoader(data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True)
    return data
