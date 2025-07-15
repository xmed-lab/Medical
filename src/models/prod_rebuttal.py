#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 16:01
# @Author  : XXXX
# @Site    : 
# @File    : doprompt.py
# @Software: PyCharm

# #Desc: prod，对应修改main_scm, config, meta_encoder.py,init.py

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
        self.domain_nums = domain_num # 不要用num_domains，其用于保证vit不变
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

        # prompt bank
        self.prompt_bank = nn.Parameter(
            torch.empty(self.domain_nums, 224//16 * 224//16, 768).normal_(std=0.02)
        , requires_grad=False)
        self.adapter = nn.Linear(768, 3)

        # # decoder design
        self.decoder = PromptDecoder(self.w, self.prompt_dim, self.head, self.trans_dim, self.z_length, self.anatomy_out_channels, self.decoder_type, self.num_mask_channels)


        self.init_weights(pretrained_vit_name, pretrained_folder)

        # # # # 参数冻结 prod全参数微调
        # for name, param in self.vision_encoder.resnet.named_parameters():
        #     # if 'mlp' not in name and 'conv1' not in name and 'layer1' not in name:
        #     param.requires_grad = False
        #
        # for name, param in self.vision_encoder.vit.named_parameters():
        #     # if 'adapter' not in name and 'norm' not in name:
        #     param.requires_grad = False # if lora suojin


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
            self.vision_encoder.resnet = load_model(self.vision_encoder.resnet,'/home/czhaobo/Medical/src/models/pretrained/resnet50_v1c.pth') 
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


    def prompt_forward(self, prompt):
        domain_cls = self.adapter(prompt)
        return domain_cls

    def domain_prompt_generate(self, encoder_out, domain_label, num_share=4, num_spec=2, lambd=0.99):
        """用CNN的产生
            每个域选几个
        """
        #share
        cnn_out = encoder_out[0] # cnn_output # B, T, D
        domain_share_prompt = (cnn_out[:num_share] + cnn_out[-num_share:]).mean(dim=0).repeat(cnn_out.shape[0],1,1) # B,T,D
        # spec
        labels_batch = torch.argmax(domain_label, dim=1).long()
        unique_labels = labels_batch.unique()
        domain_spec_prompt = torch.zeros_like(cnn_out).to(cnn_out.device)
        for i, label in enumerate(unique_labels):
            # 找出所有属于当前类别的行索引
            label_indices = (labels_batch == label).nonzero(as_tuple=True)[0]
            # 如果这个类别的行数少于2，则选择所有行；否则选择前两行
            if len(label_indices) < num_spec:
                selected_rows = cnn_out[label_indices]
            else:
                selected_rows = cnn_out[label_indices[:num_spec]]
            # 对选中的行进行加和
            domain_spec_prompt[label_indices] = selected_rows.mean(dim=0)

            # momentum update
            self.prompt_bank[label] = (1-lambd)*self.prompt_bank[label].data + lambd * selected_rows.mean(dim=0).data.detach()

        return domain_share_prompt, domain_spec_prompt


    def neutralizing(self, encoder_out, domain_label, temperature=0.5):
        """各个domain的final encoder_out需要对齐该类的domain label"""
        vit_out = encoder_out[0].mean(dim=1).squeeze().detach()
        pos_prompt = self.prompt_bank[
                domain_label.nonzero(as_tuple=False)[:, 1]].mean(dim=1).squeeze()
        neg_prompt = self.prompt_bank[
                ~domain_label.nonzero(as_tuple=False)[:, 1]].mean(dim=1).squeeze()
        pos_similarity = F.cosine_similarity(vit_out, pos_prompt)
        neg_similarity = F.cosine_similarity(vit_out, neg_prompt)
        all_similarity = torch.exp(pos_similarity + neg_similarity)
        # 计算损失
        neu_loss = -torch.log(pos_similarity/all_similarity).mean()
        return neu_loss


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

        x1 = x.repeat(1,2,1,1) # 在channel维度复制一下看看
        if config['DATA'] not in ['RVS','Pro','FUN']: # 3维度的
            x = torch.cat((x, x1), dim=1)

        if script_type == 'val' or script_type == 'test': # self.meta_loss or
            # meta_test phase
            # 寻找关系，训练adapter
            domain_cls = domain_cls_1 = neu_loss = 0
            encoder_out = self.vision_encoder(x, '0', meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                                              stop_gradient=self.stop_gradient)  # B,196,512, 本来是0

            prompt_sim = (F.normalize(encoder_out[0].mean(dim=1).squeeze()) @ F.normalize(self.prompt_bank.mean(dim=1).squeeze()).t()).unsqueeze(dim=-1).unsqueeze(dim=-1)
            domain_share = domain_spec = F.normalize((prompt_sim * self.prompt_bank.unsqueeze(0)).mean(dim=1))

            prompt_cnn = encoder_out[0] + domain_spec

            encoder_out = self.vision_encoder.mask_forward(prompt_cnn, '0')  # B,196,512, 本来是0
        else:
            # x = x + domain_prompt
            # x = torch.cat((x[:,:2,:], domain_prompt), dim=1)
            encoder_out = self.vision_encoder(x, '0', meta_loss=self.meta_loss, meta_step_size=self.meta_step_size,
                                              stop_gradient=self.stop_gradient)  # B,196,512, 本来是0
            domain_share, domain_spec = self.domain_prompt_generate(encoder_out, domain_labels)
            prompt_cnn = encoder_out[0] + F.normalize(domain_spec)
            encoder_out = self.vision_encoder.mask_forward(prompt_cnn, '0')  # B,196,512, 本来是0
            neu_loss = self.neutralizing(encoder_out, domain_labels)

            # domain_cls = self.prompt_forward(domain_share.mean(dim=1))
            domain_cls_1 = self.prompt_forward(domain_spec.mean(dim=1))


        mask = encoder_out[1]

        return mask, 0, 0, 0, domain_cls_1, (0, 0, 0), 0, 0, domain_cls_1,neu_loss


