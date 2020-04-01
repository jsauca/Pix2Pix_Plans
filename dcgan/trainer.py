import torch
import torch.optim as optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
from torch import autograd


class LR_Scheduler:

    def __init__(self, g_opt, d_opt, learning_rate_decay):
        if decay_every is None:
            self.step = self.dummy_step
        else:
            lmbda = lambda epoch: learning_rate_decay
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

    def __init__(self,
                 data,
                 generator,
                 discriminator,
                 batch_size,
                 optimizer,
                 loss_type='minimax',
                 learning_rate=None,
                 learning_rate_decay=None,
                 weight_decay=None,
                 checkpoint_disc=None,
                 checkpoint_gen=None):
        self._batch_size = batch_size
        self._data = data
        self._d_net = self._load_checkpoint(checkpoint_disc, discriminator)
        self._g_net = self._load_checkpoint(checkpoint_gen, generator)
        self._d_opt, self._g_opt = self._build_optimizer(
            optimizer, learning_rate, weight_decay)
        self._d_loss, self._g_loss = self._build_loss(loss_type)
        self._lr_scheduler = self._build_scheduler(learning_rate_decay)

    def _load_checkpoint(self, checkpoint, network):
        if use_cuda:
            network = network.cuda()
        if checkpoint is not None:
            network.load_state_dict(torch.load(checkpoint, map_location=device))
        return network

    def _build_optimizer(self, optimizer, learning_rate, weight_decay,
                         learning_rate_decay):
        if weight_decay is None:
            weight_decay = 0.
        if optimizer == 'Adam':
            optimizer_class = optim.Adam
        elif optimizer == 'SGD':
            optimizer_class = optim.SGD
        g_opt = optimizer_class(self._g_net.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        d_opt = optimizer_class(self._d_net.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        return d_opt, g_opt

    def _build_scheduler(self, learning_rate_decay):
        lr_scheduler = LR_Scheduler(self._g_opt, self._d_opt,
                                    learning_rate_decay)
        return lr_scheduler

    def _build_loss(self, loss_type):
        if loss_type == 'minimax':
            criterion = nn.BCELoss()

            def loss_function(d_x, label):
                batch_size = d_x.size(0)
                labels = torch.full((batch_size,), label, device=device)
                loss = criterion(d_x, labels)
                return loss.mean()

            def d_loss(d_real, d_fake):
                return loss_function(d_real, 1.) + loss_function(d_fake, 0.)

            def g_loss(d_fake):
                return loss_function(d_fake, 1.)
        elif loss_type == 'wasserstein':

            def gradient_penalty(x_real, x_fake):
                batch_size = x_real.size(0)
                if use_cuda:
                    alpha = torch.cuda.FloatTensor(
                        batch_size, 1, 1,
                        1).uniform_(0, 1).expand([batch_size, 3, 256, 256])
                else:
                    alpha = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(
                        0, 1).expand([batch_size, 3, 256, 256])
                x_inter = alpha * x_fake + (1 - alpha) * x_real
                x_inter.requires_grad = True
                d_inter = self._d_net(x_inter)
                gradients = autograd.grad(outputs=d_inter,
                                          inputs=x_inter,
                                          grad_outputs=torch.ones(
                                              d_inter.size()).cuda(),
                                          create_graph=True,
                                          retain_graph=True,
                                          only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
                return gradient_penalty

            def d_loss(x_real, x_fake):
        return d_loss, g_loss

    def _g_step(self):
        self._d_net.eval()
        self._g_net.train()
        x_fake = self._g_net(self._batch_size)
        d_fake = self._d_net(x_fake)
        self._g_opt.zero_grad()
        loss = self._g_loss(d_fake)
        loss.backward()
        self._g_opt.step()

    def _d_step(self, x_real):
        self._d_net.train()
        self._g_net.eval()
        x_fake = self._g_net(self._batch_size)
        d_real, d_fake = self._d_net(x_real, x_fake)
        self._d_opt.zero_grad()
        loss = self._d_loss(d_real, d_fake)
        loss.backward()
        self._d_opt.step()

    def train(self):
        for batch_idx, x_real in enumerate(self._data):
            self._d_step(x_real)
            self._g_step()

    def test(self):
        noise = torch.randn(size=(1, self._noise_size, 1, 1))
        x_fake = self._g_net(noise).squeeze().data.numpy()
        gen_image(x_fake)
