#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 15:20
# @Author  : xxx
# @Site    :
# @File    : eval_others.py
# @Software: PyCharm

# #Desc: 如何用dgnet这里也得改
import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics import dice_coeff

import numpy as np
from metrics_pro import calculate_metrics
import cv2


########################
def eval_dgnet(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    dice_tot = []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        last_name = None
        for i, sample in enumerate(loader):
            imgs, y, path = sample['data'], sample['mask'], sample['name']

            current_name = path
            if last_name is None:
                last_name = path

            imgs = imgs.to(device=device, dtype=torch.float32)
            y = y.to(dtype=mask_type)

            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= net(imgs, y, 'test')
                # reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = net(imgs, true_masks, 'test')
            mask = mask[:,:1,:,:]
            seg_output = torch.sigmoid(mask.detach().cpu())
            if current_name != last_name:  # Calculate the previous 3D volume
                dice = calculate_metrics(seg_output3D, y3D)
                dice_tot.append(dice[0])
                del seg_output3D
                del y3D

            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            last_name = current_name

            pbar.update()

        # Calculate the last 3D volume
        dice = calculate_metrics(seg_output3D, y3D)
        dice_tot.append(dice[0])

        dice_mean = np.mean(dice_tot)

    net.train()
    return dice_mean#, tot_2 / n_val,tot_3 / n_val
