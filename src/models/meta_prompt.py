#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/30 14:04
# @Author  : Anonymous
# @Site    : 
# @File    : meta_prompt.py
# @Software: PyCharm

# #Desc: global prompt
import torch
from torch import nn
from models.meta_ops import global_prompt


class PadPrompter(nn.Module):
    """
    This is for global prompt
    """
    def __init__(self, prompt_size, crop_size, channel=1):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size # padding size
        image_size = crop_size # image size
        self.channel = channel

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, channel, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, channel, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, channel, image_size-pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, channel, image_size-pad_size*2, pad_size]))

        # self.conv = nn.Conv2d(4,3, kernel_size=1, stride=1) # for main_others



    def forward(self, x, meta_step_size, meta_loss, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient
        base = torch.zeros(1, self.channel, self.base_size, self.base_size).to(x.device) 
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = global_prompt(prompt, prompt, self.meta_step_size, self.meta_loss, self.stop_gradient)
        prompt = torch.cat(x.size(0)*[prompt]) 
    
        # out = x + prompt
        out = torch.cat((x, prompt), dim=1)
        return out
    # #
    # def forward(self, x, meta_step_size, meta_loss, stop_gradient):
        # """This is for others, 3 channel"""
        # self.meta_loss = meta_loss
        # self.meta_step_size = meta_step_size
        # self.stop_gradient = stop_gradient
        # base = torch.zeros(1, self.channel, self.base_size, self.base_size).to(x.device) 
        # prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        # prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        # prompt = global_prompt(prompt, prompt, self.meta_step_size, self.meta_loss, self.stop_gradient)
        # prompt = torch.cat(x.size(0)*[prompt]) 

        # # out = prompt + x
        # out = torch.cat((x, prompt), dim=1)
        # out = self.conv(out) # for main_others
        # return out


    #




if __name__ == '__main__':
    pass