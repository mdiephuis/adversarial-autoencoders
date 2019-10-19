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
from data import *


parser = argparse.ArgumentParser(description='AAE')


# Task parametersm and model name
parser.add_argument('--uid', type=str, default='AAE',
                    help='Staging identifier (default: AAE)')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='FashionMNIST',
                    help='Name of dataset (default: FashionMNIST')

parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')


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

# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Data set transforms
transforms = [transforms.Resize((32, 32)), transforms.ToTensor()]

# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(E, D, G, E_optim, ER_optim, D_optim, G_optim, loader, train):

    data_loader = loader.train_loader if train else loader.test_loader

    # loss definitions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    E.train() if train else E.eval()
    D.train() if train else D.eval()
    G.train() if train else G.eval()

    EG_batch_loss = 0
    D_batch_loss = 0
    ER_batch_loss = 0

    for batch_idx, (x, _) in enumerate(data_loader):

        x = x.cuda() if args.cuda else x
        batch_size = x.size(0)

        if train:
            E_optim.zero_grad()
            ER_optim.zero_grad()
            D_optim.zero_grad()
            G_optim.zero_grad()

        # Encoder - Generator forward
        z_fake = E(x).view(-1, args.latent_size, 1, 1)
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
        z_real = sample_noise(batch_size, args.latent_size).view(-1, args.latent_size)
        z_real = z_real.cuda() if args.cuda else z_real

        # 2) Encoder forward, get latent z from data
        z_fake = E(x).squeeze().detach()

        # build labels for discriminator
        y_real = torch.ones(z_real.size(0), 1)
        y_fake = torch.zeros(z_fake.size(0), 1)

        y_real = y_real.cuda() if args.cuda else y_real
        y_fake = y_fake.cuda() if args.cuda else y_fake

        # Discriminator forward on sampled z_real and z_fake from encoder
        y_hat_real = D(z_real)
        y_hat_fake = D(z_fake)

        # Discriminator loss
        D_loss = bce_loss(y_hat_fake, y_fake) + bce_loss(y_hat_real, y_real)
        D_batch_loss += D_loss.item() / batch_size

        if train:
            D_loss.backward()
            D_optim.step()

        # Encoder forward, Discriminator
        z_fake = E(x).squeeze().detach()
        y_hat_fake = D(z_fake)
        ER_loss = -torch.mean(torch.log(y_hat_fake + 1e-9))
        ER_batch_loss += ER_loss.item() / batch_size

        if train:
            ER_loss.backward()
            ER_optim.step()

    # collect better stats
    return EG_batch_loss / (batch_idx + 1), D_batch_loss / (batch_idx + 1), ER_batch_loss / (batch_idx + 1)


def execute_graph(E, D, G, E_optim, ER_optim, D_optim, G_optim, loader, epoch, use_tb):

    # Training loss
    EG_t_loss, D_t_loss, ER_t_loss = train_validate(E, D, G, E_optim, ER_optim, D_optim, G_optim, loader, train=True)

    # Validation loss
    EG_v_loss, D_v_loss, ER_v_loss = train_validate(E, D, G, E_optim, ER_optim, D_optim, G_optim, loader, train=False)

    print('=> Epoch: {} Average Train EG loss: {:.4f}, D loss: {:.4f}, ER loss: {:.4f}'.format(
          epoch, EG_t_loss, D_t_loss, ER_t_loss))
    print('=> Epoch: {} Average Valid EG loss: {:.4f}, D loss: {:.4f}, ER loss: {:.4f}'.format(epoch, EG_v_loss, D_v_loss, ER_v_loss))

    if use_tb:
        logger.add_scalar(log_dir + '/EG-train-loss', EG_t_loss, epoch)
        logger.add_scalar(log_dir + '/D-train-loss', D_t_loss, epoch)
        logger.add_scalar(log_dir + '/ER-train-loss', ER_t_loss, epoch)

        logger.add_scalar(log_dir + '/EG-valid-loss', EG_v_loss, epoch)
        logger.add_scalar(log_dir + '/D-valid-loss', D_v_loss, epoch)
        logger.add_scalar(log_dir + '/ER-valid-loss', ER_v_loss, epoch)

        # # Generation examples
        img_shape = loader.img_shape[1:]

        sample = generation_example(G, args.latent_size, 10, img_shape, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('generation example', sample, epoch)


        # Reconstruction examples


    return EG_v_loss, D_v_loss, ER_v_loss


# Model definitions
E = Encoder(1, args.latent_size, 128).type(dtype)
G = Generator(1, args.latent_size, 128).type(dtype)
D = Discriminator(args.latent_size, 128).type(dtype)

# Init module weights
init_normal_weights(E, 0, 0.02)
init_normal_weights(G, 0, 0.02)
init_normal_weights(D, 0, 0.02)

# Optimiser definitions
E_optim = Adam(E.parameters(), lr=args.elr)
G_optim = Adam(G.parameters(), lr=args.glr)
D_optim = Adam(D.parameters(), lr=args.dlr)
ER_optim = Adam(E.parameters(), lr=args.erlr)


# Utils
num_epochs = args.epochs
best_loss = np.inf

# Main training loop
for epoch in range(1, num_epochs + 1):
    _, _, _ = execute_graph(E, D, G, E_optim, ER_optim, D_optim, G_optim, loader, epoch, use_tb)
