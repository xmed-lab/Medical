#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 20:56
# @Author  : Anonymous
# @Site    : 
# @File    : meta_encoder.py
# @Software: PyCharm

# #Desc: partially ref vit
import collections
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from models.resnet import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config



##################################################################



class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x, size=None):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs *x # +xs 
        else:
            x = xs
        return x


def load_pretrain(model, pre_s_dict):
    ''' Load state_dict in pre_model to models
    Solve the problem that models and pre_model have some different keys'''
    s_dict = model.state_dict()
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        if key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain models:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x



class Block(nn.Module):
    def __init__(self, dim, num_frames, num_heads, mlp_ratio=4., scale=0.5, num_tadapter=1, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapt_method=False, num_domains=1):
        super().__init__()
        self.scale = 0.5
        self.num_frames = num_frames
        self.num_tadapter = num_tadapter
        self.adapt_method = adapt_method
        self.num_domains = num_domains
        self.norm1 = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)]) if num_domains > 1 else norm_layer(
            dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.scale = scale

        if adapt_method == 'MLP':
            self.adapter1 = nn.ModuleList([Adapter(dim, skip_connect=True) for _ in range(num_domains)])
            self.adapter2 = nn.ModuleList([Adapter(dim, skip_connect=True) for _ in range(num_domains)])

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)]) if num_domains > 1 else norm_layer(
            dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, d, size=None):
        ## x in shape [BT, HW+1, D]
        B, N, D = x.shape
        int_d = int(d)
        xs = self.attn(self.norm1[int_d](x) if self.num_domains > 1 else self.norm1(x))
        if self.adapt_method:
            xs = self.adapter1[int_d](xs, size)
        x = x + self.drop_path(xs)

        xs = self.norm2[int_d](x) if self.num_domains > 1 else self.norm2(x)
        if self.adapt_method == 'ParaMLP':
            x = x + self.mlp(xs) + self.drop_path(self.scale * self.adapter2[int_d](xs, size))
        elif self.adapt_method:
            xs = self.mlp(xs)
            xs = self.adapter2[int_d](xs, size)
            x = x + self.drop_path(xs)
        else:
            xs = self.mlp(xs)
            x = x + self.drop_path(xs)
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class ViT_ImageNet(nn.Module):
    def __init__(self, img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=None, adapt_method=False,
                 num_domains=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.depth = depth
        self.num_frames = num_frames
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=patch_embedding_bias)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_frames=num_frames, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                adapt_method=adapt_method, num_domains=num_domains)
            for i in range(self.depth)]) 
        # self.ln_post = nn.ModuleList([norm_layer(embed_dim) for _ in range(num_domains)]) if self.num_domains>1 else norm_layer(embed_dim)
        # self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'temporal_embedding'}

    def forward_features(self, x, d=None):
        # B, C, H, W = x.shape
        int_d = int(d)
        # x = self.patch_embed(x)  # (B,HW,D)
        x = torch.cat(
            [self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1) 
        x = x + self.pos_embed.to(x.dtype)

        for blk in self.blocks:
            x = blk(x, d=d, size=(self.img_size // self.patch_size, self.img_size // self.patch_size))
        return x[:, 1:, :] 

    def forward(x):
        return



class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, out_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

    def forward(self, x):
        x1 = self.conv1(x) # c->n
        x2 = self.encoder1(x1) # 2n
        x3 = self.encoder2(x2) # 4n
        x4 = self.encoder3(x3) # 8n 36*36 - > 28 * 28
        return x1, x2, x3, x4


class Encoder(nn.Module):
    def __init__(self, img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=None, adapt_method=False,
                 num_domains=1):
        super(Encoder, self).__init__()
        out_channels = 128
        # self.resnet = ResNet(out_channels)
        self.resnet = resnet50(pretrained_model='/home/xxx/Medical/src/models/pretrained/resnet50_v1c.pth', norm_layer=nn.BatchNorm2d,
                                  in_channels=3,
                                  bn_eps=1e-5,
                                  bn_momentum=0.1,
                                  deep_stem=True, stem_width=64)

        self.conv_down = nn.Sequential(nn.Conv2d(8*out_channels, 768, kernel_size=3, stride=1, bias=False),
                    nn.BatchNorm2d(768),
                    nn.ReLU(inplace=True))


        self.vit = ViT_ImageNet(
            img_size=img_size, num_frames=num_frames, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            patch_embedding_bias=patch_embedding_bias, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, pretrained=pretrained,
            adapt_method=adapt_method, num_domains=num_domains)

        self.conv2 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(512))




    def forward(self, x, d=None, meta_step_size=None, meta_loss=None, stop_gradient=None):
        x1, x2, x3, x4 = self.resnet(x, meta_loss, meta_step_size, stop_gradient)
        x_down = self.conv_down(x4) # 768

        x_down = F.interpolate(x_down, size=(14, 14), mode='bilinear', align_corners=False)

        x_down = x_down.flatten(2).transpose(1, 2)
        x_trans = self.vit.forward_features(x_down, d) # 16n

        x = rearrange(x_trans, "b (x y) c -> b c x y", x=14, y=14) 
        x = self.conv2(x)
        x = F.interpolate(x, size=(config['CROP_SIZE']//16, config['CROP_SIZE']//16), mode='bilinear', align_corners=False)


        return x, x1, x2, x3






if __name__ == "__main__":
    pass


