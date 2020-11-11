import torch


def CUDA(var):
    """
    Put tensor `var` to cuda device.

    :param var: tensor
    :return: tensor
    """
    return var.cuda() if torch.cuda.is_available() else var


def CPU(var):
    """
    Put tensor `var` to cpu device.

    :param var: tensor
    :return: tensor
    """
    return var.cpu().detach()
