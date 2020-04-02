from .trainer import Trainer


def get_trainer(data, gen, disc, train_args):
    return Trainer(data, gen, disc, train_args)
