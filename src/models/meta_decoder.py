#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 16:26
# @Author  : Anonymous
# @Site    : 
# @File    : meta_decoder.py
# @Software: PyCharm

# #Desc:
from torch import Tensor

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.meta_recon import Ada_Decoder
from models.meta_ops import conv2d, linear
from models.transformer import TwoWayTransformer
from timm.models.layers import trunc_normal_
from config import config


#n =============================================



class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """I want to achieve [B,C,D]"""
        # Input projections
        q = self.q_proj(q) # B,C,D
        k = self.k_proj(k) # B,1,D
        v = self.v_proj(v) # B,1,D

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens[B,C,D,1]
        attn = attn / math.sqrt(c_per_head) # d_k
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v # B,C,D,D
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class TransHead(nn.Module):
    def __init__(self, style_dim):
        super(TransHead, self).__init__()
        dim = 8
        self.style_dim = style_dim // 2
        self.conv1 = nn.Conv2d(3, dim, 7, 1, 3, bias=True)  # dfbet对于为1
        self.conv2 = nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(dim * 4, dim * 8, 4, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(dim * 8, dim * 16, 4, 2, 1, bias=True)
        self.conv6 = nn.Conv2d(dim * 16, dim * 32, 4, 2, 1, bias=True)
        hidden = config['CROP_SIZE'] //32
        self.fc1 = nn.Linear(256 * hidden*hidden, 4 *hidden*hidden)  # 这里不是9了哦，变成8了 12(pro)，for others
        self.fc2 = nn.Linear(4 * hidden* hidden, 32)
        self.mu = nn.Linear(32, style_dim)
        self.logvar = nn.Linear(32, style_dim)

    def reparameterize(self, mu, logvar):
        """重参数"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out = linear(out.contiguous().view(-1, out.shape[1] * out.shape[2] * out.shape[3]), self.fc1.weight, self.fc1.bias, meta_loss=self.meta_loss,
                     meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=False)
        out = linear(out, self.fc2.weight, self.fc2.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
               stop_gradient=self.stop_gradient)
        # z = out
        out = F.leaky_relu(out, negative_slope=0.01, inplace=False)
        mu = linear(out, self.mu.weight, self.mu.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        logvar = linear(out, self.logvar.weight, self.logvar.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                     stop_gradient=self.stop_gradient)
        zs = self.reparameterize(mu[:,self.style_dim:], logvar[:,self.style_dim:])
        zd = self.reparameterize(mu[:,:self.style_dim], logvar[:,:self.style_dim])
        z = torch.cat((zs, zd), dim=1) 
        # mu,logvar = torch.randn(4,16), torch.randn(4,16)

        return z, mu, logvar



class SegDecoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super(SegDecoder, self).__init__()
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))
        self.recon_head = TransHead(16)

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3, raw=None, meta_loss=None, meta_step_size=0.001,
                            stop_gradient=False):

        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient
        z_out, mu, logvar = self.recon_head(raw, meta_loss=self.meta_loss, meta_step_size=0.001,
               stop_gradient=False) 

        x = self.decoder1(x, x3) 
        x = self.decoder2(x, x2) 
        x = self.decoder3(x, x1) 
        x = self.decoder4(x) # 2-n
        x = self.conv1(x)

        return x, z_out, mu, logvar


class PromptDecoder(nn.Module):
    """transunet"""
    def __init__(self, img_size, prompt_dim, head, transformer_dim, z_length,anatomy_out_channels, decoder_type, num_mask_channels):
        super(PromptDecoder, self).__init__()
        self.img_size,self.head, self.prompt_dim, self.trans_dim = img_size, head, prompt_dim, transformer_dim
        self.decoder_type, self.z_length,self.anatomy_out_channels, self.num_mask_channels = decoder_type, z_length, anatomy_out_channels, num_mask_channels

        self.attn = Attention(
            (config['CROP_SIZE']//16)*(config['CROP_SIZE']//16), 1, downsample_rate = 2
        ) 
        self.seg_head = SegDecoder(out_channels=128,  class_num=self.num_mask_channels) 
        self.recon_head = Ada_Decoder(self.decoder_type, self.anatomy_out_channels, self.z_length*2, self.num_mask_channels)



    def cross_add_local_prompt(self, x, prompt):
        b,c,w,h = x.shape
        out = x + prompt
        prompt1 = prompt.view(b, 1, -1) # B,1,H*W
        attn_out = self.attn(q=x.view(b, c, -1), k=prompt1, v=prompt1)
        # attn_out = self.attn(q=prompt.repeat(1, c, 1), k=x.view(b, c, -1), v=x.view(b, c, -1))
        attn_out = attn_out.view(b,c,w,h)
        return out + 0.001 * attn_out


    def forward(self, x, prompt,raw, meta_loss, meta_step_size, stop_gradient):
        """
        :param x: B,256,H//16,W//16,

        :param prompt: B, prompt_num, H//16,W//16
        :return:
        """
        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        src = self.cross_add_local_prompt(x[0], prompt) # B, 512, 14, 1
        # seg result
        seg_mask, z_out, mu, logvar = self.seg_head(src, x[1], x[2], x[3], raw, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                            stop_gradient=self.stop_gradient) # B, c, H//16, W//16

        seg_mask = F.interpolate(seg_mask, size=(config['CROP_SIZE'], config['CROP_SIZE']), mode='bilinear', align_corners=False)

        recon = self.recon_head(seg_mask, z_out, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                            stop_gradient=self.stop_gradient) 
        # recon = [0] # main_others

        return seg_mask, recon, z_out, mu, logvar





if __name__ == '__main__':
    pass

