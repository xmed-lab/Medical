#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 16:01
# @Author  : XXXX
# @Site    : 
# @File    : doprompt.py
# @Software: PyCharm

# #Desc: 这里是doprompt实现

# #Desc: our meta algorithm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.meta_encoder import Encoder
from models.meta_prompt import PadPrompter
from models.meta_encoder import load_pretrain
from models.resnet import load_model
from models.meta_adapter import Project
from models.meta_decoder import PromptDecoder
from models.meta_ops import conv2d
from functools import partial
from timm.models.layers import trunc_normal_
from config import config


class PromptTransUnet(nn.Module):
    def __init__(self, width, prompt_dim, head, transformer_dim, z_length, anatomy_out_channels,decoder_type, num_mask_channels,
                 prompt_size=30, domain_num=4,
                 pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
                 pretrained_folder=r'E://Code/Pycharm/Medical/src/models',
                 img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768,
                depth=12, num_heads=12, mlp_ratio=4.,
                patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                adapt_method='MLP', num_domains=1,
                     prompt_num=4,
                 **kwargs):
        super(PromptTransUnet, self).__init__()
        # 沿用的decoder传进来的
        self.w = self.h = width
        self.prompt_dim = prompt_dim
        self.trans_dim = transformer_dim
        self.head = head
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.decoder_type = decoder_type
        self.num_mask_channels = num_mask_channels


        self.prompt_size = prompt_size # global
        self.crop_size = img_size
        self.patch_size = patch_size

        self.pretrained = pretrained
        self.domain_nums = domain_num 
        self.prompt_num = prompt_num

        # global prompt
        self.global_prompt = PadPrompter(self.prompt_size, self.crop_size)

        # vision encoder
        # 先cnn试试
        self.vision_encoder = Encoder(
            img_size=224, num_frames=num_frames, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            patch_embedding_bias=patch_embedding_bias, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, pretrained=pretrained,
            adapt_method='MLP', num_domains=1) # MLP


        # adapter design
        self.adapter = Project(224//16 * (224//16)*8, self.domain_nums, img_size//16)


        # prompt bank
        # self.prompt_bank = nn.Parameter(
        #     torch.empty(self.domain_nums, self.prompt_num, 224//16, 224//16).normal_(std=0.02)
        # )
        self.prompt_bank = nn.Parameter(
           torch.empty(self.domain_nums, self.prompt_num, img_size, img_size).normal_(std=0.02))


        # # decoder design
        self.decoder = PromptDecoder(self.w, self.prompt_dim, self.head, self.trans_dim, self.z_length, self.anatomy_out_channels, self.decoder_type, self.num_mask_channels)


        self.init_weights(pretrained_vit_name, pretrained_folder)

        # # # # 参数冻结
        for name, param in self.vision_encoder.resnet.named_parameters():
            # if 'mlp' not in name and 'conv1' not in name and 'layer1' not in name:
            param.requires_grad = False

        for name, param in self.vision_encoder.vit.named_parameters():
            # if 'adapter' not in name and 'norm' not in name:
            param.requires_grad = False # if lora suojin


    def init_weights(self, pretrained_name, pretrained_folder):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.02)
                nn.init.xavier_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.pretrained:
            self.apply(_init_weights) 
            pretrained_encoder_sd = torch.load(pretrained_folder + '/pretrained/{}.pth'.format(pretrained_name))
            if pretrained_name == 'mae_pretrain_vit_base' or 'deit' in pretrained_name:
                pretrained_encoder_sd = pretrained_encoder_sd['models']
            self.vision_encoder.vit = load_pretrain(self.vision_encoder.vit, pretrained_encoder_sd)
            self.vision_encoder.resnet = load_model(self.vision_encoder.resnet,'/home/czhaobo/Medical/src/models/pretrained/resnet50_v1c.pth') # 难道不能pretrain?
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} successfully'.format(pretrained_name))
        else:
            self.apply(_init_weights)

        # for n, m in self.vision_encoder.vit.named_modules():
        #     if 'adapter' in n and 'D_fc2' in n: # self.adapt_method == 'MLP' and
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_normal_(m.weight, 0)
        #             nn.init.constant_(m.bias, 0)


    def prompt_forward(self, encoder_out, prompt_bank, meta_loss, meta_step_size, stop_gradient):
        """
        :param encoder_out: B, 256, 14, 14;
        :param prompt_bank: domain_num. prompt_num, 14, 14
        :param meta_loss:
        :param meta_step_size:
        :param stop_gradient:
        :return: domain_prompt: B, prompt_num, 14, 14; all_bias: B, domain_num
        """
        hint = encoder_out.detach() # B, 256, 14,14
        # print("Hello,", hint.shape)
        all_bias = self.adapter(hint, meta_loss, meta_step_size, stop_gradient) # B, domain_num
        all_bias = F.softmax(all_bias, dim=-1) 
        prompt_bank = prompt_bank.view(self.domain_nums, -1)
        domain_prompts = (all_bias @ prompt_bank).view(hint.shape[0], self.prompt_num, self.prompt_bank.shape[-1], self.prompt_bank.shape[-1])
        return domain_prompts, all_bias

    def forward(self, x, domain_labels, script_type, meta_loss=None, meta_step_size=config['meta_step'], stop_gradient=False): # 1e-5
        """
        :param x: B, 1, H, W
        :param domain_labels: B, domain_num
        :param script_type: train / test
        :param meta_loss: # any loss
        :param meta_step_size: lr
        :param stop_gradient:
        :return:
        """
        self.meta_loss, self.meta_step_size, self.stop_gradient = meta_loss, meta_step_size, stop_gradient

        x1 = x.repeat(1,2,1,1) 
        if config['DATA'] not in ['RVS','Pro','FUN']:
            x = torch.cat((x, x1), dim=1)
        raw = x

        if script_type == 'val' or script_type == 'test': # self.meta_loss or
            encoder_out = self.vision_encoder(x, '0', meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                                              stop_gradient=self.stop_gradient)  # B,196,512, 本来是0
            domain_prompt, domain_cls = self.prompt_forward(encoder_out[0], self.prompt_bank, None, self.meta_step_size, self.stop_gradient)
            # x = x + domain_prompt
            x = torch.cat((x[:,:2,:], domain_prompt), dim=1)
            encoder_out = self.vision_encoder(x, '0', meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                                              stop_gradient=self.stop_gradient)  # B,196,512, 本来是0
            domain_cls_1 = 0
        else:
            # 原有的
            domain_prompt = self.prompt_bank[
                domain_labels.nonzero(as_tuple=False)[:, 1]]  # [:, 1]# prompt筛选 B, prompt_num, 14, 14
            # x = x + domain_prompt
            x = torch.cat((x[:,:2,:], domain_prompt), dim=1)


            encoder_out = self.vision_encoder(x, '0', meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                                              stop_gradient=self.stop_gradient)  # B,196,512, 本来是0
            # meta_train phase
            # Phase 1
            # domain_prompt = self.prompt_bank[domain_labels.nonzero(as_tuple=False)[:, 1]] #[:, 1]# prompt筛选 B, prompt_num, 14, 14
            # x = encoder_out[0].detach() # B, 512, 14, 14
            domain_prompt, domain_cls_1 = self.prompt_forward(encoder_out[0].detach(), self.prompt_bank, None, self.meta_step_size, self.stop_gradient)
            # 生成的
            x = torch.cat((raw[:,:2,:], domain_prompt), dim=1)
            encoder_out_raw = self.vision_encoder(x, '0', meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                                              stop_gradient=self.stop_gradient)  # B,196,512, 本来是0

            all_bias = self.adapter(encoder_out_raw[0].detach(), None, self.meta_step_size, self.stop_gradient) # 这里似乎直接project过的是prompt+ encoder后； 而且其似乎不需要cls
            domain_cls = F.softmax(all_bias, dim=-1) # B, domain_num; 特征的分类

        mask = encoder_out[1]

        return mask, 0, 0, 0, domain_cls, (0, 0, 0), 0, 0, domain_cls_1


