#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:36
# @Author  : Anonymous
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# #Desc:
import torch
from matplotlib import pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if ifimg:
        image = image.transpose(1,2,0)
    return image


def kl_divergence_loss(vector1, vector2):
    return torch.distributions.kl.kl_divergence(vector1, vector2).sum()

def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()


def save_image(image, path):
    plt.imshow(image, cmap='gray') # mm no gray
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, format='pdf', dpi=600)
    plt.show()
    return 0