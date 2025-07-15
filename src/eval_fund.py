#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 20:46
# @Author  : xxxx
# @Site    : 
# @File    : eval_fund.py
# @Software: PyCharm

# #Desc:

# #Desc: 
import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics import dice_coeff

import numpy as np
from metrics_fund import calculate_metrics
import cv2


########################
def eval_dgnet(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    dice_tot_d,dice_tot_c = [], []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        last_name = None
        for i, sample in enumerate(loader):
            imgs, y, path = sample['data'], sample['mask'], sample['name']

            imgs = imgs.to(device=device, dtype=torch.float32)
            y = y.to(dtype=mask_type)

            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= net(imgs, y, 'test')
                # reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = net(imgs, true_masks, 'test')
            mask = mask[:,:2,:,:]
            seg_output = torch.sigmoid(mask.detach().cpu())
            dice_d, dice_c = calculate_metrics(seg_output, y)
            dice_tot_d.append(dice_d)
            dice_tot_c.append(dice_c)

            pbar.update()


        dice_mean_d, dice_mean_c = np.mean(dice_tot_d), np.mean(dice_tot_c)

    net.train()
    return dice_mean_d, dice_mean_c #, tot_2 / n_val,tot_3 / n_val
