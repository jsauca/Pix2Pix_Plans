import time
import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import tensorflow as tf
tf.enable_eager_execution()


def sizer():
    # desired_size =28 64/128 #first trial 28 like mnist

    imgs = []
    new_imgs = []
    path = 'images64'

    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = Image.open(os.path.join(path, f))
        img.load()
        imgs.append(img)
        fp = img.fp

    for im in imgs:
        # old_size = im.size  # old_size[0] is in (width, height) format

        #ratio = float(desired_size)/max(old_size)
        #new_size = tuple([int(x*ratio) for x in old_size])

        # images already all square and same size and ***RGB/Grey
        #im = im.resize(desired_size, Image.ANTIALIAS)

        # create a new image and paste the resized on it
        #new_im = Image.new('L', (desired_size, desired_size))
        #new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
        #new_im = np.array(im)
        # im=im.convert('L') #in grey
        new_imgs.append(np.array(im.convert('L')))
    return new_imgs


if __name__ == '__main__':
    print(sizer())
