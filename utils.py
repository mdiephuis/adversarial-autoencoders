import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def generation_example(E, G, latent_size, data_loader):

    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    draw = randn((num_class, latent_size), use_cuda)
    sample = model.decode(draw).cpu().view(num_class, 1, img_shape[0], img_shape[1])

    return sample


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


def sample_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


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
