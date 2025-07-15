#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:36
# @Author  : xxxx
# @Site    : 
# @File    : eval.py
# @Software: PyCharm

# #Desc: 如何用dgnet这里也得改
import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics import dice_coeff

def eval_dgnet(net, loader, device, mode):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_lv = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        if mode=='val':
            for imgs, true_masks, domain_labels in loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)

                with torch.no_grad():
                    mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar, domain_cls_1 = net(imgs, domain_labels, 'test')
                    # reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = net(imgs, domain_labels, 'test')
                mask_pred = mask[:, :2, :, :]


                pred = F.softmax(mask_pred, dim=1)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred[:, 0:1, :, :], true_masks[:, 0:1, :, :], device).item() # 这里的三维预测不
                tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
                pbar.update()
        else:
            for imgs, true_masks, _ in loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)

                with torch.no_grad():
                    mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= net(imgs, true_masks, 'test')
                    # reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = net(imgs, true_masks, 'test')

                mask_pred = mask[:, :2, :, :]
                pred = F.softmax(mask_pred, dim=1)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred[:, 0:1, :, :], true_masks[:, 0:1, :, :], device).item()
                tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
                pbar.update()

    net.train()
    return tot / n_val, tot_lv / n_val
