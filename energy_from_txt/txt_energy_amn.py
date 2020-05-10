
import os
import numpy as np
import torch
from floorplan_bis import getitem
from torch import nn, optim

size = 128
num_epochs = 100
channels = 17
bn = True
data_folder = '../dataset/vectors/'
use_log = False
gamma = 0.95
weight_decay = 0
freq_schedule = 10
learning_rate = 0.001
dropout = 0.5
add_rotation = True
if bn:
    dropout = 0.
activations = {
    'relu': nn.ReLU(True),
    'lrelu': nn.LeakyReLU(0.2, inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'id': nn.Identity()
}


class Conv_2D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=1,
                 activation='relu',
                 batch_norm=True,
                 transpose=False,
                 bias=False):
        super(Conv_2D, self).__init__()
        if not transpose:
            self._op = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=bias)
        else:
            self._op = nn.ConvTranspose2d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=bias)

        self._act = activations[activation]
        if batch_norm:
            self._bn = nn.BatchNorm2d(out_channels)
        else:
            self._bn = nn.Identity()
        self.init_weights(batch_norm)

    def init_weights(self, batch_norm):
        nn.init.xavier_uniform_(self._op.weight)
        if self._op.bias is not None:
            nn.init.constant_(self._op.bias, 0.0)
        if batch_norm:
            nn.init.normal_(self._bn.weight, 1.0, 0.02)
            nn.init.constant_(self._bn.bias, 0)

    def forward(self, x):
        x = self._op(x)
        x = self._bn(x)
        x = self._act(x)
        return x


def process(arr):
    if use_log:
        arr = np.log(arr)
    arr = arr.astype(float)
    mean, std = np.mean(arr), np.std(arr)
    arr -= mean
    if use_log:
        arr /= 4
    arr /= std
    return arr, mean, std


def load_corners(idx):
    path = data_folder + paths[idx] + '/'
    path += os.listdir(path)[0]
    corners = getitem(path, size, size)[0][:, :, :channels]
    corners = torch.from_numpy(corners).transpose(0, -1)
    return corners


paths, coolings, heatings = [], [], []
with open('all.csv', 'r') as reader:
    for line in list(reader)[1:1471]:
        sample = line.split(';')[1:]
        paths.append('/'.join(sample[0].split('/')))
        coolings.append(int(sample[2]))
        heatings.append(int(sample[3][:-2]))


heatings = np.array(heatings)
coolings = np.array(coolings)

num_samples = len(paths)
h, h_mean, h_std = process(heatings)
c, c_mean, c_std = process(coolings)
print('Means | Heatings {} - Coolings {}'.format(np.mean(heatings), np.mean(coolings)))
targets = np.stack([h, c], axis=1)
targets = torch.from_numpy(targets).float()
corners = list(map(load_corners, range(num_samples)))
corners = torch.stack(corners).float()
shuffle = torch.randperm(num_samples)
corners, targets = corners[shuffle], targets[shuffle]
#corners = torch.sum(corners, dim=1, keepdim=True)

x_train, y_train = corners[:-128], targets[:-128]
if add_rotation:
    x90 = x_train.transpose(2, 3)
    x180 = x_train.flip(3)
    x270 = x_train.transpose(2, 3).flip(2)
    x_train = torch.cat([x_train, x90, x180, x270], 0)
    y_train = torch.cat(4 * [y_train], 0)
dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=64, shuffle=True)
criterion = nn.MSELoss()
criterion_eval = nn.L1Loss(reduction='mean')
x_test, y_test = (corners[-128:], targets[-128:])
x_debug, y_debug = (corners[:128], targets[:128])


def criterion_real(pred, target, mean, std):
    if use_log:
        pred = torch.exp(4 * std * pred + mean)
        target = torch.exp(4 * std * target + mean)
    else:
        pred = std * pred + mean
        target = std * target + mean
    return criterion_eval(pred, target)


channels = corners.size(1)
if __name__ == '__main__':

    layers = [nn.MaxPool2d(2),
              Conv_2D(channels, 16, 3, 2, 1, 'relu', batch_norm=bn),
              nn.Dropout2d(dropout),
              nn.MaxPool2d(2),
              Conv_2D(16, 32, 3, 2, 1, 'relu', batch_norm=bn),
              nn.Dropout2d(dropout),
              nn.MaxPool2d(2),
              nn.Flatten(),
              nn.Linear(512, 256),
              nn.Dropout(dropout),
              nn.ReLU(),
              nn.Linear(256, 2)
              ]

    if use_log:
        layers.append(nn.Tanh())
    print('Data loaded')
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def lmbda(epoch): return 0.95
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lmbda)
    print('Net Loaded')

    print('*******Start Training*********')
    for epoch in range(num_epochs):
        net.train()
        loss_train = []
        for idx, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            loss_train.append(loss.item())
        loss_train = np.mean(loss_train)
        if (epoch > 0) and (epoch % freq_schedule == 0):
            scheduler.step()
        net.eval()
        x_eval, y_eval = x_debug, y_debug
        y_pred = net(x_eval)
        loss_eval_h = criterion(y_pred[:, 0], y_eval[:, 0]).item()
        loss_eval_c = criterion(y_pred[:, 1], y_eval[:, 1]).item()
        loss_real_h = criterion_real(
            y_pred[:, 0], y_eval[:, 0], h_mean, h_std).item()
        loss_real_h = int(loss_real_h)
        loss_real_c = criterion_real(
            y_pred[:, 1], y_eval[:, 1], c_mean, c_std).item()
        loss_real_c = int(loss_real_c)
        print('Epoch = {} | Loss = {:.2f}'.format(epoch + 1, loss_train))
        print('-------------------------> Train / Heatings = {:.2f} --> {} / Coolings = {:.2f} --> {}'.format(
            loss_eval_h, loss_real_h, loss_eval_c, loss_real_c))
        x_eval, y_eval = x_test, y_test
        y_pred = net(x_eval)
        loss_eval_h = criterion(y_pred[:, 0], y_eval[:, 0]).item()
        loss_eval_c = criterion(y_pred[:, 1], y_eval[:, 1]).item()
        loss_real_h = criterion_real(
            y_pred[:, 0], y_eval[:, 0], h_mean, h_std).item()
        loss_real_h = int(loss_real_h)
        loss_real_c = criterion_real(
            y_pred[:, 1], y_eval[:, 1], c_mean, c_std).item()
        loss_real_c = int(loss_real_c)
        print('-------------------------> Test / Heatings = {:.2f} --> {} / Coolings = {:.2f} --> {}'.format(
            loss_eval_h, loss_real_h, loss_eval_c, loss_real_c))
    torch.save(net, 'model')
