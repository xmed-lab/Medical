import functools
import os
import sys
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from .meta_ops import linear
import sys
sys.path.append('..')
from src.config import config

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        # prompt maker
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.linear1 = nn.Linear(128, 64)
        # self.linear2 = nn.Linear(64, 128)
        # self.linear3 = nn.Linear(256, 128)
        # self.linear4 = nn.Linear(128, 256)
        # self.linear5 = nn.Linear(512, 256)
        # self.linear6 = nn.Linear(256, 512)

        self.mlp_1= nn.Sequential(nn.Linear(128, 64),
                                 nn.GELU(),
                                  # nn.Dropout(0.3),
                                  # nn.Linear(64, 32),
                                  # nn.GELU(),
                                  # # nn.Dropout(0.3),
                                  # nn.Linear(32, 64),
                                  # nn.GELU(),
                                  # # nn.Dropout(0.3),
                                 nn.Linear(64, 128),
                                  # nn.Dropout(0.1),
                                  )
        self.mlp_2= nn.Sequential(nn.Linear(256, 128),
                                 nn.GELU(),
                                  # nn.Dropout(0.3),
                                  # nn.Linear(128, 64),
                                  # nn.GELU(),
                                  # # nn.Dropout(0.3),
                                  # nn.Linear(64, 128),
                                  # nn.GELU(),
                                  # nn.Dropout(0.3),
                                 nn.Linear(128, 256),
                                  # nn.Dropout(0.1),
                                  )
        self.mlp_3= nn.Sequential(nn.Linear(512, 256),
                                 nn.GELU(),
                                  # nn.Dropout(0.3),
                                  # nn.Linear(256, 128),
                                  # nn.GELU(),
                                  # # nn.Dropout(0.3),
                                  # nn.Linear(128, 256),
                                  # nn.GELU(),
                                  # # nn.Dropout(0.3),
                                 nn.Linear(256, 512),
                                  # nn.Dropout(0.1),
                                  )



    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x, meta_loss, meta_step_size, stop_gradient):
        blocks = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        blocks.append(x)
        x = self.maxpool(x)
        if config['adapter'] == True:
            out = self.global_avg_pooling(x).squeeze()
            prompt = F.sigmoid(self.mlp_1(out)).unsqueeze(dim=-1).unsqueeze(dim=-1)  # B,C,1,1
            x = x * prompt + x

        x = self.layer1(x);
        blocks.append(x)
        if config['adapter'] == True:
            out = self.global_avg_pooling(x).squeeze()
            prompt = F.sigmoid(self.mlp_2(out)).unsqueeze(dim=-1).unsqueeze(dim=-1)  # B,C,1,1
            x = x * prompt + x

        x = self.layer2(x);
        blocks.append(x)

        if config['adapter'] == True:
            out = self.global_avg_pooling(x).squeeze()
            prompt = F.sigmoid(self.mlp_3(out)).unsqueeze(dim=-1).unsqueeze(dim=-1)  # B,C,1,1
            x = x * prompt +x

        x = self.layer3(x);
        blocks.append(x)
        # x = self.layer4(x);
        # blocks.append(x)

        return blocks

def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    print("load resnet weight !")
    if model_file is None:
        return model

    # get the pretrained models dict
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'models' in state_dict.keys():
            state_dict = state_dict['models']
    else:
        state_dict = model_file
    t_ioend = time.time()

    # get the models dict
    model_dict = model.state_dict()

    # ini the corresponding layer
    i = 0
    j = 0
    # with open('./before.txt', 'wt') as f:
    #     print(model_dict, file=f)
    for k, v in state_dict.items():
        if k in model_dict.keys():
            if v.size() == model_dict[k].size():
                model_dict[k] = state_dict[k]
                i = i + 1
            j = j + 1
    print('total weight is',j)
    print('using weight is',i)
    # with open('./after.txt', 'wt') as f:
    #     print(model_dict, file=f)

    model.load_state_dict(model_dict, strict=False)
    ckpt_keys = set(state_dict.keys()) # pretrained keys
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    # print("Unexpected keys: {}".format(missing_keys)) 

    del state_dict
    t_end = time.time()
    print(
        "Load models, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model

def resnet18(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet34(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet50(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model



def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet152(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model




if __name__ == '__main__':
    resnet = resnet50(pretrained_model='/home/xxx/Medical/src/baselines/EPL/resnet50_v1c.pth',
                           norm_layer=nn.BatchNorm2d,
                           in_channels=1,
                           bn_eps=1e-5,
                           bn_momentum=0.1,
                           deep_stem=True, stem_width=64)

    #
    img = torch.randn(2, 1, 288, 288)
    x = resnet(img)
    print(x[0].shape, x[-1].shape)
