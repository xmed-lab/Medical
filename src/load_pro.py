#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 22:06
# @Author  :xxxx
# @Site    : 
# @File    : load_pro.py
# @Software: PyCharm

# #Desc:
import torch
from torch.utils import data
import numpy as np
import math
import os
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from torch.utils.data import DataLoader



############################## 工具函数
import os
import pandas as pd


def convert_labeled_list(root, csv_list):
    img_list = list()
    label_list = list()
    for csv_file in csv_list:
        data = pd.read_csv(os.path.join(root, csv_file))
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()
    return img_list, label_list



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






def get_train_transform(patch_size=(384, 384)):
    tr_transforms = []
    # tr_transforms.append(RandomCropTransform(crop_size=256, margins=(0, 0, 0), data_key="image", label_key="mask"))
    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="data", label_key="mask")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="data"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="data"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="data"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="data"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="data"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="data"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="data"))

    tr_transforms.append(MirrorTransform(axes=(0, 1), data_key="data", label_key="mask"))

    # now we compose these transforms together
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
    data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)

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
    data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)
    # print(data_dict['data'])
    # print(data_dict['mask'])
    # print(data_dict['name'])
    data_dict['data'], data_dict['mask'],  data_dict['reid'] = torch.from_numpy(data_dict['data']), torch.from_numpy(data_dict['mask']), None

    return data_dict


class PROSTATE_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=384, batch_size=None, img_normalize=True, reid=None):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.image_pool, self.label_pool, self.name_pool = [], [], []
        self._read_img_into_memory()
        if batch_size is not None:
            iter_nums = len(self.image_pool) // batch_size
            scale = math.ceil(250 / iter_nums)
            self.image_pool = self.image_pool * scale
            self.label_pool = self.label_pool * scale
            self.name_pool = self.name_pool * scale

        print('Image Nums:', len(self.img_list))
        print('Slice Nums:', len(self.image_pool))
        self.reid = reid

    def __len__(self):
        return len(self.image_pool)

    def __getitem__(self, item):
        img_path, slice = self.image_pool[item]
        img_sitk = sitk.ReadImage(img_path)
        img_npy = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        img_npy = self.preprocess(img_npy[slice])

        label_path, slice = self.label_pool[item]
        label_sitk = sitk.ReadImage(label_path)
        label_npy = sitk.GetArrayFromImage(label_sitk)
        label_npy = np.expand_dims(label_npy[slice], axis=0)

        if self.img_normalize:
            # img_npy = normalize_image_to_0_1(img_npy)
            for c in range(img_npy.shape[0]):
                img_npy[c] = (img_npy[c] - img_npy[c].mean()) / img_npy[c].std()
        label_npy[label_npy > 1] = 1
        return img_npy, label_npy, self.name_pool[item], self.reid

    def _read_img_into_memory(self):
        img_num = len(self.img_list)
        for index in range(img_num):
            img_file = os.path.join(self.root, self.img_list[index])
            label_file = os.path.join(self.root, self.label_list[index])

            img_sitk = sitk.ReadImage(img_file)
            label_sitk = sitk.ReadImage(label_file)

            img_npy = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
            label_npy = sitk.GetArrayFromImage(label_sitk)

            for slice in range(img_npy.shape[0]):
                if label_npy[slice, :, :].max() > 0:
                    self.image_pool.append((img_file, slice))
                    self.label_pool.append((label_file, slice))
                    self.name_pool.append(img_file)

    def preprocess(self, x):
        # x = img_npy[slice]
        mask = x > 0
        y = x[mask]

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper
        return np.expand_dims(x, axis=0)


def generate_csv():
    # 指定目录路径
    directory_path = '/home/czhaobo/Medical/data/ProDataset'  # 替换为实际目录的路径

    # 初始化存储字典，每个子目录对应一个列表
    directory_lists = {}

    # 遍历目录
    for root, dirs, files in os.walk(directory_path):
        # 初始化子目录对应的列表
        image_list = []
        mask_list = []

        for file in files:
            if file.endswith('segmentation.nii.gz') or file.endswith('Segmentation.nii.gz'):
                mask_list.append(os.path.join(root, file))
            else:
                image_list.append(os.path.join(root, file))

        # 存储到字典
        directory_lists[root] = {'image_list': image_list, 'mask_list': mask_list}

    for directory, lists in directory_lists.items():
        print(f"Directory: {directory}")
        print("Image files:")
        # for image_file in lists['image_list']:
        #     print(image_file)
        # print("\nMask files:")
        # for mask_file in lists['mask_list']:
        #     print(mask_file)
        df = pd.DataFrame({"image":lists['image_list'], "mask":lists['mask_list']})
        df.to_csv(directory + ".csv")


if __name__ == '__main__':
    # generate_csv()
    # # 测试
    config = {}
    config["ROOT"] = "/home/czhaobo/Medical/data/ProDataset/"
    config["CROP_SIZE"] = 384
    config["BATCH"] = 4
    config['Source_Dataset'] = ['UCL']
    config['TASK'] = 'BIDMC'

    source_name = config['Source_Dataset']
    source_csv = []
    for s_n in source_name:
        source_csv.append(s_n + '.csv')
    sr_img_list, sr_label_list = convert_labeled_list(config["ROOT"], source_csv)


    source_dataset = PROSTATE_dataset(config["ROOT"], sr_img_list, sr_label_list,
                                      config["CROP_SIZE"], config["BATCH"], img_normalize=False, reid=0)
    source_dataloader = DataLoader(dataset=source_dataset,
                                   batch_size=config["BATCH"],
                                   shuffle=True,
                                   pin_memory=False,
                                   collate_fn=collate_fn_w_transform,
                                   num_workers=4)

    target_test_csv = [config['TASK'] + '.csv']
    ts_img_list, ts_label_list = convert_labeled_list(config['ROOT'], target_test_csv)

    target_valid_dataset = PROSTATE_dataset(config['ROOT'], ts_img_list, ts_label_list,
                                            config['CROP_SIZE'], img_normalize=True)
    valid_loader = DataLoader(dataset=target_valid_dataset,
                              batch_size=4,
                              shuffle=False,
                              pin_memory=False,
                              collate_fn=collate_fn_wo_transform,
                              num_workers=4)

    for batch, data in enumerate(source_dataloader):
        print(data['data'])
        x, y , z= data['data'], data['mask'], data['reid'] # (4, 3, 384, 384) (4, 1, 384, 384)
        print(x.shape, y.shape)
        print(z)


    # for batch, data in enumerate(valid_loader):
    #     x, y,z = data['data'], data['mask'], data['reid'] # (4, 3, 384, 384) (4, 1, 384, 384)
    #     print(x.shape, y.shape) # (1, 3, 384, 384) (1, 1, 384, 384)
    #     print(z)

