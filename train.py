import numpy as np
import argparse
import torch
from torch.optim import Adam
from torchvision.utils import save_image

from torchvision import datasets
from torchvision import transforms

import torchvision.utils as tvu
from tensorboardX import SummaryWriter

from models import *
from utils import *


parser = argparse.ArgumentParser(description='AAE')


parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                    help='VAE latent size (default: 20')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')

# Optimizer
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of training epochs')

parser.add_argument('--elr', type=float, default=1e-3,
                    help='Encoder Learning rate (default: 1e-3')

parser.add_argument('--erlr', type=float, default=1e-4,
                    help='Encoder Learning rate (default: 1e-4')

parser.add_argument('--glr', type=float, default=1e-3,
                    help='Generator Learning rate (default: 1e-3')

parser.add_argument('--dlr', type=float, default=1e-3,
                    help='Discriminator Learning rate (default: 1e-3')


parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: logs)')


# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')


args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set visdom
use_visdom = args.visdom_url is not None

# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter()

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Get Fashion MNIST data
train_dset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
test_dset = datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True)


def train_validate(E, D, G, E_optim, E_reg_optim, D_optim, G_optim, data_loader, train):

    # loss definitions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    model.train() if train else model.eval()

    EG_batch_loss = 0
    D_batch_loss = 0
    ER_batch_loss = 0

    for batch_idx, (x, y) in enumerate(data_loader):
        x = x.cuda() if args.cuda else x
        batch_size = x.size(0)

        if train:
            E_optim.zero_grad()
            E_reg_optim.zero_grad()
            D_optim.zero_grad()
            G_optim.zero_grad()

        # Encoder - Generator forward
        z_fake = E(x)
        x_hat = G(z_fake)

        # reconstruction loss
        EG_loss = mse_loss(x_hat.view(-1, 1), x.view(-1, 1))
        EG_batch_loss += EG_loss.item() / batch_size

        if train:
            EG_loss.backward()
            G_optim.step()
            E_optim.step()

        # Discriminator forward
        # 1) sample real z
        z_real = sample_noise(batch_size, args.latent_size)
        # 2) get latent output
        z_fake = E(x)

        # build labels for discriminator
        y_real = torch.ones(z.real.size(0), 1)
        y_fake = torch.zeros(z.fake.size(0), 1)

        # Discriminator forward
        y_hat_real = D(z_real)
        y_hat_fake = D(z_fake)

        # Discriminator loss
        D_loss = bce_loss(y_hat_fake, y_fake) + bce_loss(y_hat_real, y_real)
        D_batch_loss += D_loss.item() / batch_size

        if train:
            D_loss.backward()
            D_optim.step()

        # Encoder forward, Discriminator
        z_fake = E(x)
        y_hat_fake = D(z_fake)
        ER_loss = -torch.mean(torch.log(y_hat_fake + 1e-9))
        ER_batch_loss += ER_loss.item() / batch_size

        if train:
            ER_loss.backward()
            E_reg_optim.step()

    # collect better stats
    return EG_batch_loss / (batch_idx + 1), D_batch_loss / (batch_idx + 1), ER_batch_loss / (batch_idx + 1)


def execute_graph(E, D, G, E_optim, E_reg_optim, D_optim, G_optim, train_loader, test_loader, epoch, use_tb):

    # Training loss
    t_loss = train_validate(E, D, G, E_optim, E_reg_optim, D_optim, G_optim, train_loader, train=True)

    # Validation loss
    v_loss = train_validate(E, D, G, E_optim, E_reg_optim, D_optim, G_optim, test_loader, train=False)

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
          epoch, t_loss))
    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(
          epoch, v_loss))

    if use_tb:
        pass

    return t_loss, v_loss


# Model definitions
E = Encoder(1, args.latent_size, 128)
G = Generator(1, args.latent_size, 128)
D = Discriminator(args.latent_size, 128)


# Init weights
E.apply(normal_init)
G.apply(normal_init)
D.apply(normal_init)

# Optim def
E_optim = Adam(E.parameters(), lr=args.elr)
G_optim = Adam(G.parameters(), lr=args.glr)
D_optim = Adam(D.parameters(), lr=args.dlr)
E_reg_optim = Adam(D.parameters(), lr=args.erlr)


# Util
num_epochs = args.epochs
best_loss = np.inf

# Main training loop
for epoch in range(1, num_epochs + 1):
    t_loss, v_loss = execute_graph(E, D, G, E_optim, E_reg_optim, D_optim, G_optim, train_loader, test_loader, epoch, use_tb)
