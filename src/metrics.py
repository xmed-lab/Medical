#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:35
# @Author  : Anonymous
# @Site    : 
# @File    : metrics.py
# @Software: PyCharm

# #Desc:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.spatial.distance import directed_hausdorff


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target, device):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).zero_()
        s = s.to(device)
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)




def hausdorff_distance(x, y):
    x = x.cpu().data.numpy()
    u = x.reshape(x.shape[1], -1)
    y = y.cpu().data.numpy()
    v = y.reshape(y.shape[1], -1)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class ContraLoss(nn.Module):
    def __init__(self, temp):
        super(ContraLoss, self).__init__()
        self.temp = temp

    def forward(self, emb):
        """ssl loss"""
        # emb_view1 = emb[0].view(emb[0].shape[0], -1)
        # emb_view2 = emb[-1].view(emb[-1].shape[0], -1)
        emb_view1 = emb[0]
        emb_view2 = emb[-1]
        norm_embed = F.normalize(emb_view1)
        norm_embed_1 = F.normalize(emb_view2)
        pos_score = torch.mul(norm_embed, norm_embed_1).sum(dim=1)
        pos_score = torch.exp(pos_score / self.temp)
        ttl_score = torch.matmul(norm_embed, norm_embed_1.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temp).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

