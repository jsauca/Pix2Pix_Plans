import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join


def process(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255
    shape_i = np.ones((64, 64))
    for i in range(64):
        idxs = np.where(image[i, :] < .9)[0]
        if len(idxs):
            shape_i[i, min(idxs):max(idxs) + 1] = (max(idxs) -
                                                   min(idxs) + 1) * [0.]

    shape_j = np.ones((64, 64))
    for j in range(64):
        idxs = np.where(image[:, j] < .9)[0]
        if len(idxs):
            shape_j[min(idxs):max(idxs) + 1,
                    j] = (max(idxs) - min(idxs) + 1) * [0.]

    shape = 1 - (1 - shape_i) * (1 - shape_j)
    shape = np.stack(3 * [shape], axis=2)
    image = cv2.imread(path).astype('float32') / 255

    new = np.zeros((64, 64, 3))

    indices_i, indices_j = np.where(image[:, :, 0] < 0.05)
    new[indices_i, indices_j, 0] = 1
    new[indices_i, indices_j, 1] = 1
    new[indices_i, indices_j, 2] = 0

    indices_black_i, indices_black_j = np.where(image[:, :, 1] > 0.75)

    new[indices_black_i, indices_black_j, 0] = 1
    new[indices_black_i, indices_black_j, 1] = 1
    new[indices_black_i, indices_black_j, 2] = 1

    indices_red_i, indices_red_j = np.where(
        (image[:, :, 1] < 0.9) & (image[:, :, 2] < 0.7))

    new[indices_red_i, indices_red_j, 0] = 1
    new[indices_red_i, indices_red_j, 1] = 0
    new[indices_red_i, indices_red_j, 2] = 0
    for i in range(64):
        for j in range(64):
            if sum(new[i, j, :]) == 0:
                new[i, j, :] = [1, 0, 0]
            elif sum(new[i, j, :]) == 1:
                new[i, j, :] = [0, 0, 0]

    return image, new, shape


"""Structure for torchvision loader"""
# os.makedirs('images')
# os.makedirs('images/lines')
os.makedirs('images/lines/lines')
# os.makedirs('images/shapes')
os.makedirs('images/shapes/shapes')
mypath = 'dataset/images64/'
onlyfiles = [join(mypath, f) for f in listdir(mypath)
             if isfile(join(mypath, f)) and f[-1] == 'g']

for idx, path in enumerate(onlyfiles):
    image, new, shape = process(path)
    plt.imsave('images/lines/lines/line_{}.png'.format(idx), new)
    plt.imsave('images/shapes/shapes/shape_{}.png'.format(idx), shape)
