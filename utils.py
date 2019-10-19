import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def sample_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


def generation_example(G, latent_size, n_samples, img_shape, use_cuda):

    z_real = sample_noise(n_samples, latent_size).view(-1, latent_size, 1, 1)
    z_real = z_real.cuda() if use_cuda else z_real

    x_hat = G(z_real).cpu().view(n_samples, 1, img_shape[0], img_shape[1])

    return x_hat

def reconstruct(E, G, test_loader, n_samples, img_shape, use_cuda):
    E.eval()
    G.eval()
    
    X_val, _= next(iter(test_loader))
    X_val = X_val.cuda() if use_cuda else X_val
    
    z_val = E(X_val)
    X_hat_val = G(z_val)
    
    X_val = X_val[:n_samples].cpu().view(10 * img_shape[0], img_shape[1])
    X_hat_val = X_hat_val[:n_samples].cpu().view(10 * img_shape[0], img_shape[1])
    comparison = torch.cat((X_val, X_hat_val), 1).view(10 * img_shape[0], 2 * img_shape[1])
    return comparison


def nan_check_and_break(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check_and_break(tensor, name))
    else:
        if nan_check(tensor, name) is True:
            exit(-1)


def nan_check(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check(tensor, name))
    else:
        if torch.sum(torch.isnan(tensor)) > 0:
            print("Tensor {} with shape {} was NaN.".format(name, tensor.shape))
            return True

        elif torch.sum(torch.isinf(tensor)) > 0:
            print("Tensor {} with shape {} was Inf.".format(name, tensor.shape))
            return True

    return False


def zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() > 0:
        print("tensor {} of {} dim contained ZERO!!".format(name, tensor.shape))
        exit(-1)


def all_zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() == np.prod(list(tensor.shape)):
        print("tensor {} of {} dim was all zero".format(name, tensor.shape))
        exit(-1)


def init_normal_weights(module, mu, std):
    for m in module.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(mu, std)
            m.bias.data.zero_()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_normal_weights(sub_mod, mu, std)
