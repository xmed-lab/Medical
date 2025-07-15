#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 19:03
# @Author  : Anonymous
# @Site    : 
# @File    : meta_ops.py
# @Software: PyCharm

# #Desc: base module for meta process
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


def normalization(planes, norm='in'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

def global_prompt(inputs, weight, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    inputs = inputs
    weight = weight
    if meta_loss is not None:
        if not stop_gradient: 
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]  
        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data,
                                   requires_grad=False)
        if grad_weight is not None:
            weight_adapt = weight - grad_weight * meta_step_size
        else:
            weight_adapt = weight
        return weight_adapt
    else:
        return inputs

def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    inputs = inputs
    weight = weight
    bias = bias

    if meta_loss is not None:
        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True) [0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True) [0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)




def conv2d(inputs, weight, bias, meta_step_size=0.001, stride=1, padding=0, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False, kernel_size=None):
    inputs = inputs
    weight = weight
    bias = bias


    if meta_loss is not None:
        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0] 

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                if grad_bias is not None:
                    bias_adapt = bias - grad_bias * meta_step_size
                else:
                    bias_adapt = bias
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias
        if grad_weight is not None:
            weight_adapt = weight - grad_weight * meta_step_size
        else:
            weight_adapt = weight

        return F.conv2d(inputs,
                        weight_adapt, 
                        bias_adapt, stride,
                        padding,
                        dilation, groups)
    else:
        return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)

