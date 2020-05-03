import os
import numpy as np
import torch
from floorplan_bis import getitem
from torch import nn, optim
from txt_energy import *

""" Test """
model = torch.load('model')
model.eval()

PATH = '../dataset/vectors/'
data_folder = ''

paths_test, coolings_test, heatings_test = [], [], []

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

num_samples = len(paths_test)
print(num_samples)
corners_test = list(map(lambda x: load_corners(
    x, paths_test), range(num_samples)))
print('Test data loaded')
corners_test = torch.stack(corners)

dataset_test = torch.utils.data.TensorDataset(
    corners_test.float())
test_loader = torch.utils.data.DataLoader(dataset_test)

for idx, x in enumerate(test_loader):
    y_pred = model(x)
    print(paths[idx])
    print(process.process_output(y_pred))
