
import os
import numpy as np
import torch
from floorplan_bis import getitem
from torch import nn, optim

data_folder = '../dataset/'
size = 128
channels = 17


def process(array):
    arr = np.log(array)
    mean, std = np.mean(arr), np.std(arr)
    arr -= mean
    arr /= 4 * std

    def process_output(out):
        return np.exp((out + mean) * 4 * std)
    return arr, process_output


def load_corners(idx, paths):
    path = paths[idx]
    corners = getitem(path, size, size)[0][:, :, :channels]
    corners = torch.from_numpy(corners).transpose(0, -1)
    return corners


if __name__ == '__main__':

    size = 128
    channels = 17
    paths, coolings, heatings = [], [], []
    with open('paths.csv', 'r') as reader:
        for line in list(reader)[1:]:
            sample = line.split(',')[1:]
            paths.append('/'.join(sample[0].split('/')[2:]))
            coolings.append(int(sample[1]))
            heatings.append(int(sample[2][:-2]))
    heatings = np.array(heatings)
    coolings = np.array(coolings)

    num_samples = len(paths)
    h, h_func = process(heatings)
    c, c_func = process(coolings)
    targets = np.stack([h, c], axis=1)
    targets = torch.from_numpy(targets)

    corners = list(map(load_corners, range(num_samples)))
    print('data loaded')
    corners = torch.stack(corners)

    dataset_train = torch.utils.data.TensorDataset(
        corners[:1200].float(), targets[:1200].float())
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64)

    dataset_eval = torch.utils.data.TensorDataset(
        corners[1200:].float(), targets[1200:].float())
    eval_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64)

    layers = [nn.Flatten(),
              nn.Linear(size * size * channels, 256),
              nn.ReLU(),
              nn.Linear(256, 2),
              nn.Tanh()]

    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.functional.mse_loss

    for epoch in range(10):
        for idx, (x, y) in enumerate(train_loader):
            y_pred = net(x)
            loss = criterion(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print('Train : ep = {} - it = {} - loss = {}'.format(epoch, idx, loss.item()))

        for idx, (x, y) in enumerate(eval_loader):
            y_pred = net(x)
            loss = criterion(y_pred, y)
            print('Eval : ep = {} - it = {} - loss = {}'.format(epoch, idx, loss.item()))

    torch.save(net, 'model')
