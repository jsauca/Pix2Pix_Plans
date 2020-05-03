import os
import csv
import numpy as np
import torch
from floorplan_bis import getitem
from torch import nn, optim
from txt_energy import *
""" Test """

PATH = '../dataset/vectors/'

paths_test, coolings_test, heatings_test = [], [], []

print('Test data loading...')

for path_1 in os.listdir(PATH):
    if path_1 == '_DS_Store' or path_1 == '.DS_Store':
        continue
    PATH_1 = os.path.join(PATH, path_1)
    for path_2 in os.listdir(PATH_1):
        if path_2 == '.DS_Store':
            continue
        PATH_2 = os.path.join(PATH_1, path_2)
        for path_3 in os.listdir(PATH_2):
            if path_3 == '.DS_Store':
                continue
            PATH_3 = os.path.join(PATH_2, path_3)
            txt_path = os.listdir(PATH_3)[0]
            txt_path = os.path.join(PATH_3, txt_path)
            paths_test.append(txt_path)

num_samples = len(paths_test) * 0.9

corners_test = list(map(lambda x: load_corners_test(
    x, paths_test), range(num_samples)))
print('Test data loaded - Number of samples : {}'.format(num_samples))
corners_test = torch.stack(corners_test)

h = np.zeros(num_samples)
c = np.zeros(num_samples)

targets = np.stack([h, c], axis=1)
targets = torch.from_numpy(targets)

dataset_test = torch.utils.data.TensorDataset(
    corners_test.float(), targets.float())
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
print('Data loaded')

model = torch.load('model')
model.eval()
print('Model loaded')
print('******** Start predictions *********')
heatings, coolings = [], []
with open('energy.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for idx, (x, _) in enumerate(test_loader):
        y_pred = model(x).detach().numpy()
        heating = y_pred[:, 0][0]
        cooling = y_pred[:, 1][0]
        real_heating = process_output(heating, h_mean, h_std)
        real_cooling = process_output(cooling, c_mean, c_std)
        print('File_{}_heating_{}_cooling_{}'.format(
            idx, real_heating, real_cooling))
        writer.writerow([paths_test[idx][19:-9], int(
            real_cooling), int(real_heating)])
