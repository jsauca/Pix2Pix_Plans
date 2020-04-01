from PIL import Image, ImageOps
import os
import os.path
import numpy as np

PATH = 'data/images64'
BUFFER_SIZE = 10000
BATCH_SIZE = 256  # seems have good impact **


def sizer(path=PATH):
    # desired_size =28 64/128 #first trial 28 like mnist

    imgs = []
    for f in os.listdir(path):
        print(f)
        img = Image.open(os.path.join(path, f))
        img.load()
        imgs.append(np.array(img.convert('L')))
        fp = img.fp

    return imgs


def dataset(path=PATH):
    imgs = sizer(path)
    train_images = np.array(imgs)
    train_images = train_images.reshape(train_images.shape[0], 64, 64, 1).astype(
        'float32')  # change to 1/3 if GREY/RGB
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    return train_images


if __name__ == '__main__':
    train_images = dataset()


# old_size = im.size  # old_size[0] is in (width, height) format

# ratio = float(desired_size)/max(old_size)
# new_size = tuple([int(x*ratio) for x in old_size])

# images already all square and same size and ***RGB/Grey
# im = im.resize(desired_size, Image.ANTIALIAS)

# create a new image and paste the resized on it
# new_im = Image.new('L', (desired_size, desired_size))
# new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
# new_im = np.array(im)
# im=im.convert('L') #in grey
