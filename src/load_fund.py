#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 20:06
# @Author  : J
# @Site    : 
# @File    : load_fund.py
# @Software: PyCharm

# #Desc:
import os
from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
from torch.utils.data import DataLoader


import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform, FancyColorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

import torch


def normalize_image(img_npy):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / img_npy[b, c].std()
    return img_npy


def normalize_image_to_0_1(img):
    return (img-img.min())/(img.max()-img.min())


def normalize_image_to_m1_1(img):
    return -1 + 2 * (img-img.min())/(img.max()-img.min())

def convert_labeled_list(root, csv_list):
    img_list = list()
    label_list = list()
    for csv_file in csv_list:
        data = pd.read_csv(os.path.join(root, csv_file))
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()
    return img_list, label_list


def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform((0.75, 1.25), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms




def collate_fn_w_transform(batch):
    image, label, name, reid = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    reid = np.stack(reid, 0)
    data_dict = {'data': image, 'mask': label, 'name': name, 'reid':reid}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    # 转为torch
    data_dict['data'], data_dict['mask'],  data_dict['reid'] = torch.from_numpy(normalize_image(data_dict['data'])), torch.from_numpy(data_dict['mask']), torch.from_numpy(data_dict['reid'])

    return data_dict


def collate_fn_wo_transform(batch):
    image, label, name, reid = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    reid = np.stack(reid, 0)
    data_dict = {'data': image, 'mask': label, 'name': name, 'reid':reid}
    data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    data_dict['data'], data_dict['mask'],  data_dict['reid'] = torch.from_numpy(data_dict['data']), torch.from_numpy(data_dict['mask']), None
    return data_dict


def to_one_hot_list(mask_list):
    list = []
    for i in range(mask_list.shape[0]):
        mask = to_one_hot(mask_list[i].squeeze(0))
        list.append(mask)
    return np.stack(list, 0)


def to_one_hot(pre_mask, classes=2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [1, 0]
    mask[pre_mask == 2] = [1, 1]
    return mask.transpose(2, 0, 1)



###################################################### loader


class OPTIC_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=512, img_normalize=True, reid=None):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.reid = reid

        for i in range(len(self.label_list)): # v依然放在下面会g
            if self.label_list[i].endswith('tif'):
                self.label_list[i] = self.label_list[i].replace('.tif', '-{}.tif'.format(1))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        # if self.label_list[item].endswith('tif'): # 会更改self list
        #     self.labels_list[item] = self.labels_list[item].replace('.tif', '-{}.tif'.format(1))
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file).convert('L')

        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            # img_npy = normalize_image_to_0_1(img_npy)
            for c in range(img_npy.shape[0]):
                img_npy[c] = (img_npy[c] - img_npy[c].mean()) / img_npy[c].std()
        label_npy = np.array(label)

        mask = np.zeros_like(label_npy)
        mask[label_npy < 255] = 1 # 注意！！！！
        mask[label_npy == 0] = 2
        return img_npy, mask[np.newaxis], img_file, self.reid


if __name__ == '__main__':
    config = {}
    config["ROOT"] = "/home/czhaobo/Medical/data/FundusDataset/"
    config["CROP_SIZE"] = 512
    config["BATCH"] = 4
    config['TASK'] = 'Magrabia'

    source_name = config['TASK']
    source_csv = [config["TASK"] + '_test.csv', config["TASK"] + '_train.csv']
    sr_img_list, sr_label_list = convert_labeled_list(config["ROOT"], source_csv)
    source_dataset = OPTIC_dataset(config['ROOT'], sr_img_list, sr_label_list,
                                   config['CROP_SIZE'], img_normalize=False, reid=[0])
    source_dataloader = DataLoader(dataset=source_dataset,
                                   batch_size=config["BATCH"],
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=collate_fn_w_transform,
                                   num_workers=2)

    source_name = config['TASK']
    source_csv = [config["TASK"] + '_test.csv', config["TASK"] + '_train.csv']
    sr_img_list, sr_label_list = convert_labeled_list(config["ROOT"], source_csv)
    source_dataset = OPTIC_dataset(config['ROOT'], sr_img_list, sr_label_list,
                                   config['CROP_SIZE'], img_normalize=False, reid=[0])
    source_dataloader2 = DataLoader(dataset=source_dataset,
                                   batch_size=config["BATCH"],
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=collate_fn_w_transform,
                                   num_workers=2)


    target_test_csv = [config["TASK"] + '_test.csv', config["TASK"] + '_train.csv']
    ts_img_list, ts_label_list = convert_labeled_list(config["ROOT"], target_test_csv)



    target_valid_dataset = OPTIC_dataset(config['ROOT'], ts_img_list, ts_label_list,
                                         config["CROP_SIZE"], img_normalize=True)
    test_dataloader = DataLoader(dataset=target_valid_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn_wo_transform,
                                 num_workers=2)


    for batch, data in enumerate(source_dataloader):
        print(data['mask'].shape)
        print(data.keys())
        x, y , z= data['data'], data['mask'], data['reid'] # (4, 3, 384, 384) (4, 2, 384, 384)
        print(data['mask'])
        print(data['mask'][0,0,:,:].sum()) # 是大的
        print(data['mask'][0,1,:,:].sum()) # 是小的，里面更小的部分 # 具有包含关系
        print((data['mask'][0,0,:,:].long() & data['mask'][0,1,:,:].long()).sum())
        print("XXXXXXXXXXXXXXXXXX")
        # print(x.shape, y.shape)
        # print(z)

    # for batch, data in enumerate(source_dataloader):
    #     x, y , z = data['data'], data['mask'], data['reid'] # (4, 3, 512, 512) (4, 2, 384, 384)
    #     print(x.shape, y.shape)
