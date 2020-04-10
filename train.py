import dataset, dcgan, options
import torch
torch.backends.cudnn.enabled = False
### Options
args = options.get_train_args()

### Data
data = dataset.get_dataset(args)

### Discriminator
disc = dcgan.get_discriminator(args)

### Generator
gen = dcgan.get_generator(args)

### Trainer
trainer = dcgan.get_trainer(data, gen, disc, args)
for epoch in range(1000):
    trainer.train()
    trainer.save_checkpoints()
    samples = trainer.test()
    trainer.save_samples(samples)
