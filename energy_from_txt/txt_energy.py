
import os
import numpy as np
import torch
from floorplan_bis import getitem
from torch import nn, optim


def process(array):
    arr = np.log(array)
    mean, std = np.mean(arr), np.std(arr)
    arr -= mean
    arr /= 4 * std
    return arr, mean, std


def process_output(out, mean, std):
    return np.exp(out * 4 * std + mean)


def load_corners(idx):
    path = data_folder + paths[idx] + '/'
    path += os.listdir(path)[0]
    corners = getitem(path, size, size)[0][:, :, :channels]
    corners = torch.from_numpy(corners).transpose(0, -1)
    return corners


def load_corners_test(idx, paths):
    path = paths[idx]
    corners = getitem(path, size, size)[0][:, :, :channels]
    corners = torch.from_numpy(corners).transpose(0, -1)
    return corners


size = 128
channels = 17
data_folder = '../dataset/'

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
h, h_mean, h_std = process(heatings)
c, c_mean, c_std = process(coolings)
targets = np.stack([h, c], axis=1)
targets = torch.from_numpy(targets)

if __name__ == '__main__':

    corners = list(map(load_corners, range(num_samples)))
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

    print('Data loaded')
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.functional.mse_loss
    print('Net Loaded')

    print('*******Start Training*********')
    for epoch in range(10):
        for idx, (x, y) in enumerate(train_loader):
            y_pred = net(x)
            y_bis = y_pred.detach().numpy()
            heating = y_bis[:, 0][0]
            cooling = y_bis[:, 1][0]
            print('File_{}_heating_{}_cooling_{}'.format(idx, process_output(heating, h_mean, h_std),
                                                         process_output(cooling, c_mean, c_std)))
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
