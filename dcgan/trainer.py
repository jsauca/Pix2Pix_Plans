import torch
import torch.optim as optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()


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
        self._lr_scheduler = self._build_scheduler(learning_rate_decay)

    def _load_checkpoint(self, checkpoint, network):
        if use_cuda:
            device = 'gpu'
            network = network.cuda()
        else:
            device = 'cpu'
        if checkpoint is not None:
            network.load_state_dict(
                torch.load(checkpoint, map_location=torch.device(device)))
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

    def _g_step(self):

        noise = torch.randn(size=(self._batch_size, self._noise_size, 1, 1))
        labels = torch.ones((self._batch_size, 1), dtype=int)
        x_fake = self._g_net(noise)
        d_fake = self._d_net(x_fake).unsqueeze(1)
        self._g_opt.zero_grad()
        loss = nn.BCELoss()(d_fake.float(), labels.float())
        loss.backward()
        self._g_opt.step()

    def _d_step(self, x_real):
        noise = torch.randn(size=(self._batch_size // 2, self._noise_size, 1,
                                  1)).float()
        labels_zeros = torch.zeros((self._batch_size // 2, 1),
                                   dtype=int).float()
        labels_ones = torch.ones((self._batch_size // 2, 1), dtype=int).float()
        x_fake = self._g_net(noise)
        d_fake = self._d_net(x_fake.float())
        loss_fake = nn.BCELoss()(d_fake.unsqueeze(1), labels_zeros)
        x_real = x_real.double()
        d_real = self._d_net(x_real.float())

        loss_real = nn.BCELoss()(d_real.unsqueeze(1), labels_ones)
        loss = loss_real + loss_fake
        self._d_opt.zero_grad()
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
