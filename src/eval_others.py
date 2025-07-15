#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 15:20
# @Author  : xxxx
# @Site    : 
# @File    : eval_others.py
# @Software: PyCharm

# #Desc:
import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics import dice_coeff
from torchmetrics.classification import F1Score


def eval_dgnet(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    f1_score = F1Score(task="multiclass", num_classes=2, multidim_average ='samplewise')
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:

        for i, sample in enumerate(loader):
            imgs, true_masks = sample['image'], sample['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= net(imgs, true_masks, 'test')

            mask_pred = mask[:, :1, :, :]

            # seg_pred = torch.sigmoid(mask_pred)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            dsc = dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
            # dsc = f1_score(torch.stack([1 - seg_pred[:, 0], seg_pred[:, 0]], dim=1), true_masks[:, 0].long())[1]
            tot += dsc

            pbar.update()

    net.train()
    return tot / n_val
