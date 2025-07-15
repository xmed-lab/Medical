#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:35
# @Author  : Anonymous
# @Site    : 
# @File    : losses.py
# @Software: PyCharm

# #Desc: loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff



def beta_vae_loss(reco_x, x, logvar, mu, beta, type='L2'):
    if type == 'BCE':
        reco_x_loss = F.binary_cross_entropy(reco_x, x, reduction='sum')
    elif type == 'L1':
        reco_x_loss = F.l1_loss(reco_x, x, size_average=False)
    else:
        reco_x_loss = F.mse_loss(reco_x, x, size_average=False)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (reco_x_loss + beta * kld) / x.shape[0]


def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()



# def dice_loss_pro(pred, target):
    # """
    # This definition generalize to real valued pred and target vector.
    # This should be differentiable.
    # pred: tensor with first dimension as batch
    # target: tensor with first dimension as batch
    # """
    # smooth = 1  # 1e-12

    # # have to use contiguous since they may from a torch.view op
    # iflat = pred.contiguous().view(-1) # https://zhuanlan.zhihu.com/p/64376950
    # tflat = target.contiguous().view(-1)
    # intersection = (iflat * tflat).sum()

    # # A_sum = torch.sum(tflat * iflat)
    # # B_sum = torch.sum(tflat * tflat)
    # loss = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean() # (D, )

    # return 1 - loss

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.1  # 1e-12

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1) # https://zhuanlan.zhihu.com/p/64376950
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    # A_sum = torch.sum(tflat * iflat)
    # B_sum = torch.sum(tflat * tflat)
    loss = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean() # (D, )

    return 1 - loss


def HSIC_lossfunc(x, y):
    assert x.dim() == y.dim() == 2
    assert x.size(0) == y.size(0)
    m = x.size(0)
    h = torch.eye(m) - 1 / m
    h = h.to(x.device)
    K_x = gaussian_kernel(x)
    K_y = gaussian_kernel(y)
    return K_x.mm(h).mm(K_y).mm(h).trace() / (m - 1 + 1e-10)


def gaussian_kernel(x, y=None, sigma=5):
    if y is None:
        y = x
    assert x.dim() == y.dim() == 2
    assert x.size() == y.size()
    z = ((x.unsqueeze(0) - y.unsqueeze(1)) ** 2).sum(-1)
    return torch.exp(- 0.5 * z / (sigma * sigma))


def LS_dis(score_real, score_fake):
    return 0.5 * (torch.mean((score_real - 1) ** 2) + torch.mean(score_fake ** 2))


def LS_model(score_fake):
    return 0.5 * (torch.mean((score_fake - 1) ** 2))



""" Full assembly of the parts to form the complete network """
def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN metrics for the discriminator.

    Inputs:
    - scores_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - metrics: A PyTorch Variable containing the metrics.
    """
    loss = (torch.mean((scores_real - 1) ** 2) + torch.mean(scores_fake ** 2)) / 2
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN metrics for the generator.

    Inputs:
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - metrics: A PyTorch Variable containing the metrics.
    """
    loss = torch.mean((scores_fake - 1) ** 2) / 2
    return loss


# %%

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy metrics function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE metrics over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


# %%

def discriminator_loss(logits_real, logits_fake, device):
    """
    Computes the discriminator metrics described above.

    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - metrics: PyTorch Variable containing (scalar) the metrics for the discriminator.
    """
    true_labels = torch.ones(logits_real.size()).to(device=device, dtype=torch.float32)
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, true_labels - 1)
    return loss


def generator_loss(logits_fake, device):
    """
    Computes the generator metrics described above.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - metrics: PyTorch Variable containing the (scalar) metrics for the generator.
    """
    true_labels = torch.ones(logits_fake.size()).to(device=device, dtype=torch.float32)
    loss = bce_loss(logits_fake, true_labels)
    return loss




def hausdorff_distance(x, y):
    x = x.cpu().data.numpy()
    u = x.reshape(x.shape[1], -1)
    y = y.cpu().data.numpy()
    v = y.reshape(y.shape[1], -1)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])