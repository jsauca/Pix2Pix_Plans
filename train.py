import dataset
import pix2pix
import options
import torch
#torch.backends.cudnn.enabled = False
# Options
args = options.get_train_args()

# Data
data = dataset.get_dataset(args)

# Discriminator
disc = pix2pix.get_discriminator(args)

# Generator
gen = pix2pix.get_generator(args)

# Trainer
trainer = pix2pix.get_trainer(data, gen, disc, args)
for epoch in range(500):
    trainer.train()
    trainer.save_checkpoints()
    samples = trainer.test()
    trainer.save_samples(samples)
    trainer.show_samples(samples)
    # trainer.show_animation()
