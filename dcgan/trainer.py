import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
from torch import autograd


class LR_Scheduler:

    def __init__(self, d_opt, g_opt, learning_rate_decay):
        if learning_rate_decay is None or learning_rate_decay == 0.:
            self.step = self.dummy_step
        else:
            lr_lambda = lambda epoch: learning_rate_decay
            self.g_scheduler = optim.lr_scheduler.MultiplicativeLR(
                g_opt, lr_lambda)
            self.d_scheduler = optim.lr_scheduler.MultiplicativeLR(
                d_opt, lr_lambda)
            self.step = self.decay_step

    def dummy_step(self):
        return None

    def decay_step(self):
        self.g_scheduler.step()
        self.d_scheduler.step()


class Trainer:

    def __init__(self, data, gen, disc, args):
        self._args = args
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

    def _build_dir(self):
        self._dir = os.path.join(os.getcwd(), 'temp',
                                 datetime.now().strftime('%m-%d_%H-%M-%S'))

    def _build_optimizer(self):
        if self._args.optimizer == 'adam':
            optimizer_class = optim.Adam
        elif self._args.optimizer == 'sgd':
            optimizer_class = optim.SGD
        self._g_opt = optimizer_class(self._g_net.parameters(),
                                      lr=self._args.learning_rate,
                                      weight_decay=self._args.weight_decay)
        self._d_opt = optimizer_class(self._d_net.parameters(),
                                      lr=self._args.learning_rate,
                                      weight_decay=self._args.weight_decay)

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
                    alpha = torch.cuda.FloatTensor(
                        batch_size, 1, 1,
                        1).uniform_(0, 1).expand([batch_size, 3, 256, 256])
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
                loss = -d_real.mean(0) + d_fake.mean(0)
                loss += gradient_penalty(x_real, x_fake)
                return loss

            def g_loss(d_fake):
                return -d_fake.mean(0)

        self._d_loss = d_loss
        self._g_loss = g_loss

    def _g_step(self):
        self._d_net.eval()
        self._g_net.train()
        x_fake = self._g_net(self._args.batch_size)
        d_fake = self._d_net(x_fake)
        self._g_opt.zero_grad()
        loss = self._g_loss(d_fake)
        loss.backward()
        self._g_opt.step()

    def _d_step(self, x_real):
        self._d_net.train()
        self._g_net.eval()
        x_fake = self._g_net(self._args.batch_size)
        d_real, d_fake = self._d_net(x_real, x_fake)
        self._d_opt.zero_grad()
        loss = self._d_loss(d_real, d_fake, x_real, x_fake)
        loss.backward()
        self._d_opt.step()

    def train(self):
        self._epoch += 1
        for batch_idx, (x_real, _) in enumerate(self._data):
            print(batch_idx)
            self._d_step(x_real)
            if np.random.uniform() < self._args.gen_prob:
                self._g_step()
        self.lr_scheduler.step()
        epoch_dir = self._dir + '/epoch_{}'.format(self._epoch)
        os.makedirs(epoch_dir)
        ckp_dir = epoch_dir + '/checkpoint.pt'
        torch.save(model.state_dict(), ckp_dir)
