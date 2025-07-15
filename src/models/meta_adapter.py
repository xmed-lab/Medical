#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 10:07
# @Author  : Anonymous
# @Site    : 
# @File    : meta_adapter.py
# @Software: PyCharm

# #Desc:
import torch.nn as nn
import torch.nn.functional as F
from models.meta_ops import linear
from timm.models.layers import trunc_normal_



class Project(nn.Module):
    """
    input: B，prompt_num 14, 14
    """
    def __init__(self, n_inputs, n_outputs, mlp_width) -> None:
        super().__init__()
        self.input = nn.Linear(n_inputs, n_inputs//2)
        self.hidden = nn.Linear(n_inputs//2, n_inputs // 8)
        self.hidden2 = nn.Linear(n_inputs//8, n_inputs // 16)
        self.output = nn.Linear(n_inputs // 16, n_outputs)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient
        x = x[:,:8,:,:]
        b,_,_,_ =  x.shape

        x = x.reshape(b, -1)
        if meta_loss is not None:
            x = linear(x, self.input.weight, self.input.bias, meta_loss=self.meta_loss,
                         meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
            x = F.relu(x)
            x = linear(x, self.hidden.weight, self.hidden.bias, meta_loss=self.meta_loss,
                         meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
            x = F.relu(x)
            x = linear(x, self.hidden2.weight, self.hidden2.bias, meta_loss=self.meta_loss,
                         meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
            x = F.relu(x)
            x = linear(x, self.output.weight, self.output.bias, meta_loss=self.meta_loss,
                         meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        else:
            x = self.input(x)
            x = F.relu(x)
            x = self.hidden(x)
            x = F.relu(x)
            x = self.hidden2(x)
            x = F.relu(x)
            x = self.output(x)

        return x