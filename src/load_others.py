#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 13:15
# @Author  : xxxx
# @Site    : 
# @File    : load_others.py
# @Software: PyCharm

# #Desc: other data


import torch
from torch.utils.data import Dataset
from imageio import imread
from data_others import get_dg_segtransform, train_collate_fn, \
    train_dg_collate_fn, test_collate_fn, test_dg_collate_fn
import os
import random

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 phase='train',
                 splitid=[2, 3, 4],
                 transform=None,
                 state='train',
                 reid=None,
                 ):
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.label_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.img_name_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        self.reid = reid

        SEED = 1023
        random.seed(SEED)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})

        self.transforms = transform
        self._read_img_into_memory()
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        # Display stats
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': self.reid}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': self.reid}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample=anco_sample
        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            if self.splitid[0] == 4:
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((144, 144, 144+512, 144+512)).resize((256, 256), Image.LANCZOS))
                _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
                _target = _target[144:144+512, 144:144+512]
                _target = Image.fromarray(_target)
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                _target = Image.open(self.image_list[index]['label'])

            if _target.mode == 'RGB':
                _target = _target.convert('L')
            if self.state != 'prediction':
                _target = _target.resize((256, 256))
            self.label_pool[Flag].append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)



class RetinalVesselSegmentation(Dataset):
    """
    Retinal vessel segmentation dataset
    including 4 domain datasets
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 phase='train',
                 splitid=[0,1,2,3],# [1, 2, 3],
                 transform=None,
                 state='train',
                 normalize='whitening',
                 reid=None,
                 ):
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.normalize = normalize
        self.image_list = []
        self.phase = phase
        self.image_pool = {'CHASEDB1': [], 'DRIVE': [], 'HRF': [], 'STARE': []}
        self.label_pool = {'CHASEDB1': [], 'DRIVE': [], 'HRF': [], 'STARE': []}
        self.roi_pool = {'CHASEDB1': [], 'DRIVE': [], 'HRF': [], 'STARE': []}
        self.img_name_pool = {'CHASEDB1': [], 'DRIVE': [], 'HRF': [], 'STARE': []}
        self.postfix = [['jpg', 'png', 'png'], ['tif', 'tif', 'gif'], ['jpg', 'tif', 'tif'], ['ppm', 'ppm', 'png']]
        self.splitid = splitid
        self.reid = reid


        SEED = 1023
        random.seed(SEED)
        domain_dirs = os.listdir(self._base_dir)
        domain_dirs.sort()
        for i in range(len(domain_dirs)):
            domain_dirs[i] = os.path.join(self._base_dir, domain_dirs[i])

        for id in splitid: # 选取用于训练的数据集？
            if id == 3:
                self._image_dir = domain_dirs[id]
            else:
                self._image_dir = os.path.join(domain_dirs[id], phase) # STARE没有test train这一说法
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))
            # 不同的数据类型，看起来有点复杂啊
            postfix_im, postfix_gt, postfix_roi = self.postfix[id][0], self.postfix[id][1], self.postfix[id][2]
            imagePath = glob(os.path.join(self._image_dir, 'image', '*.{}'.format(postfix_im)))
            gtPath = glob(os.path.join(self._image_dir, 'mask', '*.{}'.format(postfix_gt)))
            roiPath = glob(os.path.join(self._image_dir, 'roi', '*.{}'.format(postfix_roi)))
            imagePath.sort()
            gtPath.sort()
            roiPath.sort()

            if id == 3 and phase != 'test':
                imagePath, gtPath, roiPath = imagePath[:10], gtPath[:10], roiPath[:10] # 选前10个训练
            elif id == 3 and phase == 'test':
                imagePath, gtPath, roiPath = imagePath[10:], gtPath[10:], roiPath[10:] # 选后10个进行测试

            for idx, image_path in enumerate(imagePath):
                self.image_list.append({'image': image_path, 'label': gtPath[idx], 'roi': roiPath[idx]})

        self.transforms = transform
        self._read_img_into_memory() # 读取数据
        iterFlag = True
        while iterFlag:
            iterFlag = False
            for key in self.image_pool: # 三个域
                if len(self.image_pool[key]) < 1:
                    print('key ' + key + ' has no data')
                    del self.image_pool[key]
                    del self.label_pool[key]
                    del self.roi_pool[key]
                    del self.img_name_pool[key]
                    iterFlag = True
                    break

        # Display stats
        for key in self.image_pool:
            print('{} images in {}'.format(len(self.image_pool[key]), key))
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        """dataloader中会调用"""
        max = -1
        for key in self.image_pool:
            if len(self.image_pool[key]) > max:
                max = len(self.image_pool[key])
        if self.phase != 'test': # 为啥要放大三倍啊？嗷，这里原来是3个域
            # max *= 3
            max = max
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                # anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': self.reid}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _roi = self.roi_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'roi': _roi, 'img_name': _img_name, 'dc': domain_code}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample = anco_sample

        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = self.image_list[index]['image']
            if 'DRIVE' in basename:
                Flag = 'DRIVE'
            elif 'CHASEDB1' in basename:
                Flag = 'CHASEDB1'
            elif 'STARE' in basename:
                Flag = 'STARE'
            elif 'HRF' in basename:
                Flag = 'HRF'

            if Flag == 'STARE':
                im = imread(self.image_list[index]['image'])
                gt = imread(self.image_list[index]['label'])
                self.image_pool[Flag].append(Image.fromarray(im).convert('RGB').resize((512, 512), Image.LANCZOS))
                self.label_pool[Flag].append(Image.fromarray(gt).convert('L').resize((512, 512)))
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((512, 512), Image.LANCZOS))
                self.label_pool[Flag].append(
                    Image.open(self.image_list[index]['label']).convert('L').resize((512, 512)))

            self.roi_pool[Flag].append(
                Image.open(self.image_list[index]['roi']).convert('L').resize((512, 512)))

            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)

        print('img_num: ' + str(img_num))



############################################## dataloader

def get_seg_dg_dataloader(cfg, batch_size, workers, reid=None):
    """reid指重定向id"""
    transform_train, transform_test = get_dg_segtransform(cfg['DATASET'])
    dataroot = cfg['ROOT']

    if cfg['DATASET'] == 'optic':
        trainset = FundusSegmentation(dataroot, phase='train', splitid=cfg['TRAIN'], transform=transform_train,reid=reid)
        testset = FundusSegmentation(dataroot, phase='test', splitid=cfg['TEST'], transform=transform_test, reid=reid)
    if cfg['DATASET'] == 'rvs':
        trainset = RetinalVesselSegmentation(dataroot, phase='train', splitid=cfg['TRAIN'], transform=transform_train, reid=reid)
        testset = RetinalVesselSegmentation(dataroot, phase='test', splitid=cfg['TEST'], transform=transform_test, reid=reid)

    # print("XXXXXXXXXXXXXXXXX", len(trainset))

    train_sampler = None

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #             trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers,
        pin_memory=False, drop_last=True, collate_fn=train_dg_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size if cfg['DATASET'] in ['optic', 'rvs'] else 1, shuffle=False, num_workers=workers, pin_memory=False,
        drop_last=False, collate_fn=test_dg_collate_fn if cfg['DATASET'] in ['optic', 'rvs'] else train_dg_collate_fn
    )

    return trainset, train_loader, test_loader


if __name__ == '__main__':
    # RetinalVesselSegmentation('../data/RetinalDataset/')
    cfg = {}

    cfg['DATASET'] = 'rvs'
    cfg['ROOT'] = '/home/czhaobo/Medical/data/RetinalDataset'
    cfg['TRAIN'] = [1]
    cfg['TEST'] = [2]

    trainset, train_loader, test_loader = get_seg_dg_dataloader(cfg, batch_size=4, workers=4, reid=1) # 一个batch为4， test loader对应两个batch
    print(len(train_loader)) # 10->28; 15->44
    # for i, sample in enumerate(train_loader):
    #     print(sample['image'].shape)
    #     print(sample['label'].shape) # mask
        # print(sample['dc'])
        # print(sample['roi'].shape)
    #
    for i, sample in enumerate(test_loader):
        print(sample['image'].shape)
        print(sample['label'].shape) # mask 256 * 256
        # print(sum(sample['roi']==sample['label']))





