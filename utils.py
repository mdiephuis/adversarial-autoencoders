import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def sample_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


def pca_project(x, num_elem=2):

    if isinstance(x, torch.Tensor) and len(x.size()) == 3:
        batch_proj = []
        for batch_ind in range(x.size(0)):
            tensor_proj = pca_project(x[batch_ind].squeeze(0), num_elem)
            batch_proj.append(tensor_proj)
        return torch.cat(batch_proj)

    xm = x - torch.mean(x, 1, keepdim=True)
    xx = torch.matmul(xm, torch.transpose(xm, 0, -1))
    u, s, _ = torch.svd(xx)
    x_proj = torch.matmul(u[:, 0:num_elem], torch.diag(s[0:num_elem]))
    return x_proj


def latentspace2d_example(E, img_shape, n_samples, use_cuda):
    E.eval()
    num_x, num_y = 20, 20
    latent_size = 2

    x_values = np.linspace(-3, -3, num_x)
    y_values = np.linspace(-3, -3, num_y)

    canvas = np.empty((num_y * img_shape[0], num_y * img_shape[1]))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            draw = torch.from_numpy(
                np.array([[np.float(xi), np.float(yi)]] * n_samples))

            draw = draw.view(-1, 2, 1, 1)
            # draw = draw.view(-1, 1, 1, 1)
            draw = draw.cuda() if use_cuda else draw

            x_hat = E(draw).cpu().detach().numpy()
            x_hat = x_hat[0].reshape(img_shape[0], img_shape[1])

            canvas[(num_x - i - 1) * img_shape[1]:(num_x - i) * img_shape[1],
                   j * img_shape[1]:(j + 1) * img_shape[1]] = x_hat

    return canvas


def latentcluster2d_example(E, model_type, data_loader, use_pca, use_cuda):
    E.eval()
    img_shape = data_loader.img_shape[1:]

    data = []
    labels = []
    for _, (x, y) in enumerate(data_loader.test_loader):
        x = x.cuda() if use_cuda else x
        if model_type != 'conv':
            x = x.view(-1, img_shape[0] * img_shape[1])

        z = E(x)
        data.append(z.detach().cpu())
        y = y.detach().cpu().numpy()
        labels.extend(y.flatten())

    centroids = torch.cat(data)
    centroids = centroids.reshape(-1, z.size(1))

    if centroids.size(1) > 2 and use_pca:
        centroids = pca_project(centroids, 2)
    elif centroids.size(1) > 2:
        centroids = centroids[:, :2]

    return centroids.numpy(), labels


def generation_example(G, model_type, latent_size, n_samples, img_shape, use_cuda):

    z_real = sample_noise(n_samples, latent_size).view(-1, latent_size, 1, 1)
    z_real = z_real.cuda() if use_cuda else z_real

    if model_type != 'conv':
        z_real = z_real.view(-1, latent_size)

    x_hat = G(z_real).cpu().view(n_samples, 1, img_shape[0], img_shape[1])

    return x_hat


def reconstruct(E, G, model_type, test_loader, n_samples, img_shape, use_cuda):
    E.eval()
    G.eval()

    x, _ = next(iter(test_loader))
    x = x.cuda() if use_cuda else x

    if model_type != 'conv':
        x = x.view(-1, img_shape[0] * img_shape[1])

    z_val = E(x)

    x_hat = G(z_val)

    x = x[:n_samples].cpu().view(10 * img_shape[0], img_shape[1])
    x_hat = x_hat[:n_samples].cpu().view(10 * img_shape[0], img_shape[1])
    comparison = torch.cat((x, x_hat), 1).view(10 * img_shape[0], 2 * img_shape[1])
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
