import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import torchvision.utils as tvu
from torch.autograd import Variable
import matplotlib.pyplot as plt


def normal_init(module, mu, std):
    for m in module.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(mu, std)
            m.bias.data.zero_()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                normal_init(sub_mod)


def sample_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)
