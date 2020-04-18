import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class LR_Scheduler:

    def __init__(self, d_opt, g_opt, learning_rate_decay):
        if learning_rate_decay is None or learning_rate_decay == 0.:
            self.step = self.dummy_step
        else:

            def lr_lambda(epoch):
                return learning_rate_decay

            self._g_scheduler = optim.lr_scheduler.MultiplicativeLR(
                g_opt, lr_lambda)
            self._d_scheduler = optim.lr_scheduler.MultiplicativeLR(
                d_opt, lr_lambda)
            self.step = self.decay_step

    def dummy_step(self):
        return None

    def decay_step(self):
        self._g_scheduler.step()
        self._d_scheduler.step()


class Trainer:

    def __init__(self, data, gen, disc, args):
        print('* Building trainer ...')
        self._args = args
        if args.conditional:
            self._data = data[0]
            self._sampler = data[1]
        else:
            self._data = data
        self._d_net = disc.to(device)
        self._g_net = gen.to(device)
        self._build_dir()
        self._init_train()
        self._build_optimizer()
        self._build_loss()
        self._build_scheduler()

    def _init_train(self):
        self._epoch = 0
        self._samples = []

    def _build_dir(self):
        self._dir = os.path.join(os.getcwd(), 'temp',
                                 datetime.now().strftime('%m-%d_%H-%M-%S'))
        print('----> Creating directory = {}'.format(self._dir))
        os.makedirs(self._dir)

    def _build_optimizer(self):
        if self._args.optimizer == 'adam':
            optimizer_class = optim.Adam
        elif self._args.optimizer == 'sgd':
            optimizer_class = optim.SGD
        self._g_opt = optimizer_class(self._g_net.parameters(),
                                      lr=self._args.learning_rate,
                                      weight_decay=self._args.weight_decay,
                                      betas=(0.5, 0.999))
        self._d_opt = optimizer_class(self._d_net.parameters(),
                                      lr=self._args.learning_rate,
                                      weight_decay=self._args.weight_decay,
                                      betas=(0.5, 0.999))

    def _build_scheduler(self):
        self._lr_scheduler = LR_Scheduler(self._d_opt, self._g_opt,
                                          self._args.learning_rate_decay)

    def _build_loss(self):
        if self._args.loss_type == 'minimax':
            criterion = nn.BCELoss()

            def loss_function(d_x, label):
                batch_size = d_x.size(0)
                labels = torch.full((batch_size,), label, device=device)
                loss = criterion(torch.sigmoid(d_x), labels)
                return loss.mean(0)

            def d_loss(d_real, d_fake, x_real, x_fake):
                return loss_function(d_real, 1.) + loss_function(d_fake, 0.)

            def g_loss(d_fake):
                return loss_function(d_fake, 1.)

        elif self._args.loss_type == 'wasserstein':

            def gradient_penalty(x_real, x_fake):
                batch_size = x_real.size(0)
                if use_cuda:
                    alpha = torch.cuda.FloatTensor(batch_size, 1, 1,
                                                   1).uniform_(0, 1).expand(
                                                       [batch_size, 3, 64, 64])
                else:
                    alpha = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(
                        0, 1).expand([batch_size, 3, 64, 64])
                x_inter = (1 - alpha) * x_real + alpha * x_fake
                x_inter.requires_grad = True
                d_inter = self._d_net(x_inter)
                gradients = autograd.grad(outputs=d_inter,
                                          inputs=x_inter,
                                          grad_outputs=torch.ones(
                                              d_inter.size()).to(device),
                                          create_graph=True,
                                          retain_graph=True,
                                          only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
                return gradient_penalty

            def d_loss(d_real, d_fake, x_real, x_fake):
                loss = -d_real.mean() + d_fake.mean()
                loss += 10 * gradient_penalty(x_real, x_fake)
                return loss

            def g_loss(d_fake):
                return -d_fake.mean()

        self._d_loss = d_loss
        self._g_loss = g_loss

    def _g_step(self, condition=None):
        self._g_opt.zero_grad()
        # self._d_net.eval()
        # self._g_net.train()
        if self._args.conditional:
            x_fake = self._g_net(condition, True)
            d_fake = self._d_net(torch.cat([x_fake, condition], 1))
        else:
            x_fake = self._g_net(self._args.batch_size, True)
            d_fake = self._d_net(x_fake)
        loss = self._g_loss(d_fake)
        loss.backward()
        self._g_opt.step()

    def _d_step(self, x_real, condition=None):
        self._d_opt.zero_grad()
        # self._d_net.train()
        # self._g_net.eval()
        if self._args.conditional:
            x_fake = self._g_net(condition).detach()
            cgan_x_fake = torch.cat([x_fake, condition], 1)
            cgan_x_real = torch.cat([x_real, condition], 1)
            d_fake = self._d_net(cgan_x_fake)
            d_real = self._d_net(cgan_x_real)
            loss = self._d_loss(d_real, d_fake, cgan_x_real, cgan_x_fake)
        else:
            x_fake = self._g_net(self._args.batch_size).detach()
            d_real, d_fake = self._d_net(x_real, x_fake)
            loss = self._d_loss(d_real, d_fake, x_real, x_fake)
        loss.backward()
        self._d_opt.step()

    def train(self):
        self._epoch += 1
        if self._epoch == 1:
            print('* Begin training ...')
        print('--> Training epoch = {} ...'.format(self._epoch))
        for batch_idx, sample in tqdm(enumerate(self._data)):
            if self._args.conditional:

                x_real = sample[0][0].to(device)
                condition = sample[1][0].to(device)
                self._cdt = condition
            else:
                x_real = sample[0]
                condition = None

            self._d_step(x_real, condition)
            if np.random.uniform() < self._args.gen_prob:
                self._g_step(condition)
            if batch_idx > 3:
                break
        self._lr_scheduler.step()
        self._epoch_dir = self._dir + '/epoch_{}'.format(self._epoch)
        print('--> Training epoch = {} done !'.format(self._epoch))
        os.makedirs(self._epoch_dir)
        self._epoch_dir += '/'

    def save_checkpoints(self, d_save=True, g_save=True):
        if d_save and not self._args.debug:
            path = self._epoch_dir + 'disc_checkpoint.pt'
            print('--> Saving discriminator checkpoint = {} ...'.format(path))
            torch.save(self._d_net.state_dict(), path)
        if g_save and not self._args.debug:
            path = self._epoch_dir + 'gen_checkpoint.pt'
            print('--> Saving generator checkpoint = {} ...'.format(path))
            torch.save(self._g_net.state_dict(), path)

    def test(self, to_numpy=False):
        print('--> Generating {} samples ...'.format(self._args.num_samples))
        with torch.no_grad():
            if self._args.conditional:
                condition = self._cdt
                samples = self._g_net(condition) * 0.5 + 0.5
                return condition, samples
            else:
                samples = self._g_net(self._args.num_samples) * 0.5 + 0.5
                return samples

    def save_samples(self, samples, nrow=8, normalize=True, padding=2):
        condition = None
        if type(samples) == tuple:
            condition, samples = samples
        print('--> Saving samples = {}'.format(self._epoch_dir))
        vutils.save_image(samples,
                          self._epoch_dir + 'grid.png',
                          normalize=normalize,
                          padding=padding)
        if condition is not None:
            vutils.save_image(condition,
                              self._epoch_dir + 'condition_grid.png',
                              normalize=normalize,
                              padding=padding)
        if not self._args.debug:
            for sample_idx, sample in enumerate(samples):
                vutils.save_image(
                    sample,
                    self._epoch_dir + 'sample_{}.png'.format(sample_idx))

    def show_samples(self, samples, nrow=8, normalize=False, padding=2):

        def plot(x):
            i = vutils.make_grid(x, normalize=normalize, padding=padding)
            i = np.transpose(i.cpu().detach(), (1, 2, 0))
            fig = plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.imshow(i)
            plt.show()

        if type(samples) == tuple:
            plot(samples[0])
            plot(samples[1])
        else:
            plot(samples)
        #self._samples.append(i)
        # plt.clf()

    # def show_animation(self):
    #     from IPython.display import HTML
    #     fig = plt.figure(figsize=(8, 8))
    #     plt.axis("off")
    #     ims = [[plt.imshow(i, animated=True)] for i in self._samples]
    #     ani = animation.ArtistAnimation(fig,
    #                                     ims,
    #                                     interval=1000,
    #                                     repeat_delay=1000,
    #                                     blit=True)
    #     print('blaaaaaaaaaaaaaaaaaa')
    #     HTML(ani.to_html5_video())
    #     # plt.clf()
