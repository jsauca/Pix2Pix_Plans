from data import get_dataset
from dcgan import get_discriminator, get_generator
from procedures import get_trainer
from options import get_train_args

args = get_train_args()
data = get_dataset(args)
disc = get_discriminator(args)
gen = get_generator(args)
trainer = get_trainer(data, gen, disc, args)
trainer.train()
