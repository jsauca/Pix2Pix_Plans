import dataset
import dcgan
import options
import torch
#torch.backends.cudnn.enabled = False
# Options
args = options.get_train_args()

# Data
data = dataset.get_dataset(args)

# Discriminator
disc = dcgan.get_discriminator(args)

# Generator
gen = dcgan.get_generator(args)
