#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:38
# @Author  : xxxx
# @Site    :
# @File    : loader.py
# @Software: PyCharm

# #Desc: data loader
import random
import os
import math
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from config import config


# Data directories
LabeledVendorA_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorA/'
LabeledVendorA_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorA/'

LabeledVendorB_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorB/'
LabeledVendorB_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorB/'

LabeledVendorC_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorC/'
LabeledVendorC_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorC/'

LabeledVendorD_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorD/'
LabeledVendorD_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorD/'


UnlabeledVendorA_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorA/'
UnlabeledVendorB_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorB/'
UnlabeledVendorC_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorC/'
UnlabeledVendorD_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorD/'


Labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB_data_dir, LabeledVendorC_data_dir, LabeledVendorD_data_dir]
Labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB_mask_dir, LabeledVendorC_mask_dir, LabeledVendorD_mask_dir]
Unlabeled_data_dir = [UnlabeledVendorA_data_dir, UnlabeledVendorB_data_dir, UnlabeledVendorC_data_dir, UnlabeledVendorD_data_dir]






def prepare_inference_loader(batch_size, test_vendor, image_size=224):
    """
    没搞清楚为啥这里batch_sz传进来要除以二
    :param batch_size:
    :param test_vendor:
    :param image_size:
    :return:
    """
    random.seed(14)
    LabeledVendorA_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorA/'
    LabeledVendorA_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorA/'

    LabeledVendorB_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorB/'
    LabeledVendorB_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorB/'

    LabeledVendorC_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorC/'
    LabeledVendorC_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorC/'

    LabeledVendorD_data_dir = '../data/SCMDataset/ready/scgm_split_2D_data/Labeled/vendorD/'
    LabeledVendorD_mask_dir = '../data/SCMDataset/ready/scgm_split_2D_mask/Labeled/vendorD/'

    UnlabeledVendorA_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorA/'
    UnlabeledVendorB_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorB/'
    UnlabeledVendorC_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorC/'
    UnlabeledVendorD_data_dir = '../data/SCMDataset/scgm_split_2D_data/Unlabeled/vendorD/'

    Labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB_data_dir, LabeledVendorC_data_dir,
                        LabeledVendorD_data_dir]
    Labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB_mask_dir, LabeledVendorC_mask_dir,
                        LabeledVendorD_mask_dir]
    Unlabeled_data_dir = [UnlabeledVendorA_data_dir, UnlabeledVendorB_data_dir, UnlabeledVendorC_data_dir,
                          UnlabeledVendorD_data_dir]
    test_loader = prepare_inference_folder(Labeled_data_dir, Labeled_mask_dir, batch_size, image_size, test_num=test_vendor)
    return test_loader


def prepare_inference_folder(data_folders, mask_folders, batch_size, new_size=288, test_num='D', num_workers=1):
    print("Current test domains: ", test_num)
    if test_num == 'A':  # meta-test: A domain
        domain_1_img_dirs = [data_folders[1]]
        domain_1_mask_dirs = [mask_folders[1]]
        domain_2_img_dirs = [data_folders[2]]
        domain_2_mask_dirs = [mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[0]]
        test_mask_dirs = [mask_folders[0]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    elif test_num == 'B':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[2]]
        domain_2_mask_dirs = [mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[1]]
        test_mask_dirs = [mask_folders[1]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]


    elif test_num == 'C':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1]]
        domain_2_mask_dirs = [mask_folders[1]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[2]]
        test_mask_dirs = [mask_folders[2]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    elif test_num == 'D':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1]]
        domain_2_mask_dirs = [mask_folders[1]]
        domain_3_img_dirs = [data_folders[2]]
        domain_3_mask_dirs = [mask_folders[2]]

        test_data_dirs = [data_folders[3]]
        test_mask_dirs = [mask_folders[3]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    else:
        print('Wrong test vendor!')
    test_dataset = ImageFolder(test_data_dirs, test_mask_dirs, train=False, labeled=True)
    print("QQQQ", len(test_dataset))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False,
                             num_workers=num_workers, pin_memory=False)  # A设定10；

    return test_loader



def get_meta_split_data_loaders(batch_size, test_vendor='D', image_size=224):
    """
    没搞清楚为啥这里batch_sz传进来要除以二
    :param batch_size:
    :param test_vendor:
    :param image_size:
    :return:
    """
    random.seed(14)

    domain_1_labeled_loader, domain_1_unlabeled_loader, \
    domain_2_labeled_loader, domain_2_unlabeled_loader,\
    domain_3_labeled_loader, domain_3_unlabeled_loader, \
    test_loader, \
    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset = \
        get_data_loader_folder(Labeled_data_dir, Labeled_mask_dir, batch_size, image_size, test_num=test_vendor)

    return domain_1_labeled_loader, domain_1_unlabeled_loader, \
    domain_2_labeled_loader, domain_2_unlabeled_loader,\
    domain_3_labeled_loader, domain_3_unlabeled_loader, \
    test_loader, \
    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset


def get_data_loader_folder(data_folders, mask_folders, batch_size, new_size=288, test_num='D', num_workers=1):
    print("Current test domains: ", test_num)
    if test_num=='A': # meta-test: A domain
        domain_1_img_dirs = [data_folders[1]]
        domain_1_mask_dirs = [mask_folders[1]]
        domain_2_img_dirs = [data_folders[2]]
        domain_2_mask_dirs = [mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[0]]
        test_mask_dirs = [mask_folders[0]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    elif test_num=='B':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[2]]
        domain_2_mask_dirs = [mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[1]]
        test_mask_dirs = [mask_folders[1]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]


    elif test_num=='C':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1]]
        domain_2_mask_dirs = [mask_folders[1]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[2]]
        test_mask_dirs = [mask_folders[2]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    elif test_num=='D':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1]]
        domain_2_mask_dirs = [mask_folders[1]]
        domain_3_img_dirs = [data_folders[2]]
        domain_3_mask_dirs = [mask_folders[2]]

        test_data_dirs = [data_folders[3]]
        test_mask_dirs = [mask_folders[3]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    else:
        print('Wrong test vendor!')

    domain_1_labeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, label=0, num_label=domain_1_num, train=True, labeled=True)
    domain_2_labeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, label=1, num_label=domain_2_num, train=True, labeled=True)
    domain_3_labeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, label=2, num_label=domain_3_num, train=True, labeled=True)

    # analysis需要, 还需要把test_loader的drop设置为True
    domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)
    domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)
    domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)

    # domain_1_labeled_loader = None
    # domain_2_labeled_loader = None
    # domain_3_labeled_loader = None

    # https://blog.csdn.net/weixin_40123108/article/details/85099449: 就是transform后的dataset
    domain_1_unlabeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, label=0, train=True, labeled=False) # num_label和labeled是同步的
    domain_2_unlabeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, label=1, train=True, labeled=False)
    domain_3_unlabeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, label=2, train=True, labeled=False)

    # 这里数据增强一下
    cutmix_mask_prop_range = (0.25, 0.5)
    cutmix_boxmask_n_boxes = 1
    cutmix_boxmask_fixed_aspect_ratio = False
    cutmix_boxmask_by_size = False
    cutmix_boxmask_outside_bounds = False
    cutmix_boxmask_no_invert = False
    from models.mask_gen import BoxMaskGenerator, AddMaskParamsToBatch,SegCollate
    mask_generator = BoxMaskGenerator(prop_range=cutmix_mask_prop_range, n_boxes=cutmix_boxmask_n_boxes,
                                      random_aspect_ratio=not cutmix_boxmask_fixed_aspect_ratio,
                                      prop_by_area=not cutmix_boxmask_by_size,
                                      within_bounds=not cutmix_boxmask_outside_bounds,
                                      invert=not cutmix_boxmask_no_invert)
    add_mask_params_to_batch = AddMaskParamsToBatch(
        mask_generator
    )
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    domain_1_unlabeled_loader = DataLoader(dataset=domain_1_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=False )
    domain_2_unlabeled_loader = DataLoader(dataset=domain_2_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=False )
    domain_3_unlabeled_loader = DataLoader(dataset=domain_3_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=False)


    test_dataset = ImageFolder(test_data_dirs, test_mask_dirs, train=False, labeled=True)
    print("QQQQ",len(test_dataset))
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=False) # A设定10；

    return domain_1_labeled_loader, domain_1_unlabeled_dataset, \
                domain_2_labeled_loader, domain_2_unlabeled_dataset,\
                domain_3_labeled_loader, domain_3_unlabeled_dataset, \
                 test_loader, \
           domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset





def get_data_loader_folder2(data_folders, mask_folders, batch_size, new_size=288, test_num='D', num_workers=1):
    print("Current test domains: ", test_num)
    if test_num=='A': # meta-test: A domain
        domain_1_img_dirs = [data_folders[1]]
        domain_1_mask_dirs = [mask_folders[1]]
        domain_2_img_dirs = [data_folders[2]]
        domain_2_mask_dirs = [mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[0]]
        test_mask_dirs = [mask_folders[0]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    elif test_num=='B':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[2]]
        domain_2_mask_dirs = [mask_folders[2]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[1]]
        test_mask_dirs = [mask_folders[1]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]


    elif test_num=='C':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1]]
        domain_2_mask_dirs = [mask_folders[1]]
        domain_3_img_dirs = [data_folders[3]]
        domain_3_mask_dirs = [mask_folders[3]]

        test_data_dirs = [data_folders[2]]
        test_mask_dirs = [mask_folders[2]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    elif test_num=='D':
        domain_1_img_dirs = [data_folders[0]]
        domain_1_mask_dirs = [mask_folders[0]]
        domain_2_img_dirs = [data_folders[1]]
        domain_2_mask_dirs = [mask_folders[1]]
        domain_3_img_dirs = [data_folders[2]]
        domain_3_mask_dirs = [mask_folders[2]]

        test_data_dirs = [data_folders[3]]
        test_mask_dirs = [mask_folders[3]]

        domain_1_num = [10]
        domain_2_num = [10]
        domain_3_num = [10]
        test_num = [10]

    else:
        print('Wrong test vendor!')

    domain_1_labeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, label=0, num_label=domain_1_num, train=True, labeled=True)
    domain_2_labeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, label=1, num_label=domain_2_num, train=True, labeled=True)
    domain_3_labeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, label=2, num_label=domain_3_num, train=True, labeled=True)


    # domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    # domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    # domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    domain_1_labeled_loader = None
    domain_2_labeled_loader = None
    domain_3_labeled_loader = None

    # https://blog.csdn.net/weixin_40123108/article/details/85099449: 就是transform后的dataset
    domain_1_unlabeled_dataset = ImageFolder(domain_1_img_dirs, domain_1_mask_dirs, label=0, train=True, labeled=False) # num_label和labeled是同步的
    domain_2_unlabeled_dataset = ImageFolder(domain_2_img_dirs, domain_2_mask_dirs, label=1, train=True, labeled=False)
    domain_3_unlabeled_dataset = ImageFolder(domain_3_img_dirs, domain_3_mask_dirs, label=2, train=True, labeled=False)



    test_dataset = ImageFolder(test_data_dirs, test_mask_dirs, train=False, labeled=True)

    return domain_1_labeled_loader, domain_1_unlabeled_dataset, \
                domain_2_labeled_loader, domain_2_unlabeled_dataset,\
                domain_3_labeled_loader, domain_3_unlabeled_dataset, \
                 test_dataset, \
           domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset




def default_loader(path):
    """load image"""
    return np.load(path)['arr_0']

def make_dataset(dir):
    """return image path list"""
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

class ImageFolder(data.Dataset):
    def __init__(self, data_dirs, mask_dirs, train=True, label=None,num_label=None, labeled=True, loader=default_loader):
        print('Loading Domain: ', data_dirs)

        temp_imgs, temp_masks = [], []
        domain_labels = []

        if train:
            # 20%
            k = config['RATIO']  # 用5%的有label数据
        else:
            k = 1

        for num_set in range(len(data_dirs)):  # 这里其实都可以去掉
            data_roots = sorted(make_dataset(data_dirs[num_set]))  # image data path list
            mask_roots = sorted(make_dataset(mask_dirs[num_set]))
            num_label_data = 0
            for num_data in range(len(data_roots)):  # 遍历这个image list
                if labeled:
                    if train:
                        
                        n_label = str(math.ceil(num_label[num_set] * k + 1))  # cut point; 这里是图片对应的个数，比如'30'
                        if '00' + n_label == data_roots[num_data][-10:-7] or '0' + n_label == data_roots[num_data][
                                                                                              -10:-7]:  # 前三位是过滤病人，这里取5%的病人作为有label的
                            # print(n_label) # 仅用这几个数据，其他都是无标签数据
                            # print(data_roots[num_data][-10:-7])
                            break

                    for num_mask in range(len(mask_roots)):  # labeled data
                        if data_roots[num_data][-10:-4] == mask_roots[num_mask][-10:-4]:  # 保证标签和图片一致
                            temp_imgs.append(data_roots[num_data])
                            temp_masks.append(mask_roots[num_mask])
                            domain_labels.append(label)
                            num_label_data += 1
                        else:
                            pass
                else:
                    temp_imgs.append(data_roots[num_data])
                    domain_labels.append(label)


        imgs = temp_imgs
        masks = temp_masks
        labels = domain_labels

        self.imgs = imgs
        self.masks = masks
        self.labels = labels  # domain label
        self.new_size = config['CROP_SIZE']
        self.loader = loader
        self.labeled = labeled  # true or false
        self.train = train

    def __getitem__(self, index):
        if self.train:
            index = random.randrange(len(self.imgs))
        else:
            pass

        path_img = self.imgs[index]
        img = self.loader(path_img)
        img = Image.fromarray(img)
        h, w = img.size

        label = self.labels[index]

        if label==0: # domain label
            one_hot_label = torch.tensor([[1], [0], [0]])
        elif label==1:
            one_hot_label = torch.tensor([[0], [1], [0]])
        elif label==2:
            one_hot_label = torch.tensor([[0], [0], [1]])
        else:
            one_hot_label = torch.tensor([[0], [0], [0]])

        # Augmentations:
        if self.labeled:
            path_mask = self.masks[index]
            mask = self.loader(path_mask)
            mask = mask[:, :, 1]
            mask = Image.fromarray(mask)
            if self.train:
                # rotate, random angle between 0 - 90
                angle = random.randint(0, 90)
                img = F.rotate(img, angle, InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, InterpolationMode.NEAREST)

                ## crop
                if h > 110 and w > 110:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.new_size, self.new_size))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)

                img = transform(img)
                mask = transform(mask)

                img = F.to_tensor(np.array(img))
                mask = F.to_tensor(np.array(mask))
                mask = (mask > 0.1).float()
                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0) # 最后一层是bg

                return img, mask, one_hot_label.squeeze() # pytorch: N,C,H,W; 需要对mask和image做完全相同的操作这个dataset会加上one-hot

            else: # test的预处理方式不一样
                if h > 110 and w > 110:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.new_size, self.new_size))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)
                img = transform(img)
                mask = transform(mask)

                img = F.to_tensor(np.array(img))
                mask = F.to_tensor(np.array(mask))
                mask = (mask > 0.1).float()
                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0) # 最后一层是bg

                return img, mask, path_img # 最后返回的居然只有img和mask,要多一个path img

        else: # unlabel的处理方式不一样
            # rotate, random angle between 0 - 90
            angle = random.randint(0, 90)
            img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            if h > 110 and w > 110:
                size = (100, 100)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = [transforms.Resize((self.new_size, self.new_size))] + transform_list
                transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                transform = transforms.Compose(transform_list)
            else:
                size = (100, 100)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                transform = transforms.Compose(transform_list)

            img = transform(img)
            img = F.to_tensor(np.array(img))

            return img, one_hot_label.squeeze()  # pytorch: N,C,H,W

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    test_vendor = 'D'
    domain_1_labeled_loader, domain_1_unlabeled_loader, \
    domain_2_labeled_loader, domain_2_unlabeled_loader, \
    domain_3_labeled_loader, domain_3_unlabeled_loader, \
    test_loader, \
    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset = get_meta_split_data_loaders(batch_size=2,test_vendor=test_vendor)

    from torch.utils.data import DataLoader, ConcatDataset

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset])

    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, drop_last=True, pin_memory=True,
                            num_workers=2)
    dataiter = iter(val_loader)
    output = next(dataiter)
    print(output[0].shape, output[1].shape)
