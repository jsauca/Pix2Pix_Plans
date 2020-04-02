import argparse

# DCGAN Model
from dcgan.generator import *
from dcgan.discriminator import *
from dcgan.trainer import *
from dcgan.utils import *

# Settings
parser = argparse.ArgumentParser()
## Parameters for networks
parser.add_argument('--channels',
                    help='number of channels for generated images',
                    type=int,
                    default=3)
### Parameters for discriminator
parser.add_argument('--disc_version',
                    help='version for discriminator',
                    type=int,
                    default=0)
parser.add_argument('--disc_scale',
                    help='scale of hidden dimensions',
                    type=int,
                    default=64)
parser.add_argument('--disc_checkpoint',
                    help='checkpoint for discriminator',
                    type=str,
                    default=None)
### Parameters for generator
parser.add_argument('--gen_version',
                    help='version for discriminator',
                    type=int,
                    default=0)
parser.add_argument('--gen_scale',
                    help='scale of hidden dimensions',
                    type=int,
                    default=64)
parser.add_argument('--gen_checkpoint',
                    help='checkpoint for discriminator',
                    type=str,
                    default=None)
parser.add_argument('--noise_size',
                    help='size of latent space',
                    type=int,
                    default=100)

## Parameters for trainer
parser.add_argument('--data_folder',
                    help='folder for training data',
                    type=str,
                    default='data/')
parser.add_argument('--batch_size',
                    help='batch size for training',
                    type=int,
                    default=4)
parser.add_argument('--gen_prob',
                    help='probability/frequency of training generator',
                    type=float,
                    default=0.2)
parser.add_argument('--optimizer',
                    help='optimizer type : adam, sgd or rms',
                    type=str,
                    default='adam')
parser.add_argument('--loss_type',
                    help='loss for model : wasserstein or minimax',
                    type=str,
                    default='wasserstein')
parser.add_argument('--learning_rate',
                    help='learning rate for optimizer',
                    type=float,
                    default=1e-3)
parser.add_argument('--learning_rate_decay',
                    help='learning rate decay for optimizer',
                    type=float,
                    default=1.)
parser.add_argument('--weight_decay',
                    help='weight decay for optimizer',
                    type=float,
                    default=0.)

parser.add_argument('--checkpoint_gen',
                    help='checkpoint for generator',
                    type=str,
                    default=None)
args = parser.parse_args()

data = get_dataset(args)
disc = get_discriminator(args)
gen = get_generator(args)
trainer = Trainer(data, gen, disc, args)
trainer.train()
