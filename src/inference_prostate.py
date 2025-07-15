#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 15:57
# @Author  : xxxx
# @Site    : 
# @File    : inference_prostate.py
# @Software: PyCharm

# #Desc: 这里放一些prostate的
from tqdm import tqdm
import torch
import os
import cv2
import numpy as np
from load_pro import normalize_image,convert_labeled_list, PROSTATE_dataset, collate_fn_wo_transform, collate_fn_w_transform
from config import config
from torch.utils.data import DataLoader
from metrics_pro import calculate_metrics
# import np.bool_ as np.bool
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


def save_images(model_path, loader, device, result_path):
    model = torch.load(model_path).to(device)
    print('model load sucess')

    dice_tot, asd_tot = [], []
    mask_type = torch.float32
    nval = len(loader)
    model.eval()
    with tqdm(total=nval, desc='Validation round', unit='batch', leave=False) as pbar:
        last_name = None
        for i, sample in enumerate(loader):
            imgs, y, path = sample['data'], sample['mask'], sample['name']

            current_name = path
            if last_name is None:
                last_name = path

            imgs = imgs.to(device=device, dtype=torch.float32)
            y = y.to(dtype=mask_type)

            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets, mu, logvar, domain_cls_1 = model(imgs, y, 'test')
                # reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = net(imgs, true_masks, 'test')
            mask = mask[:, :1, :, :]
            seg_output = torch.sigmoid(mask.detach().cpu())
            if current_name != last_name:  # Calculate the previous 3D volume
                dice, asd = calculate_metrics(seg_output3D, y3D)
                dice_tot.append(dice)
                asd_tot.append(asd)

                del seg_output3D
                del y3D

            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            # draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
            # cv2.imwrite(result_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_img.png',
            #             imgs[0].detach().cpu().numpy().transpose(1,2,0) * 255)
            # cv2.imwrite(result_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_mask.png',
            #             (y[0][0].detach().cpu().numpy() * 255).astype(np.uint8))
            # cv2.imwrite(result_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_pred.png',
            #             draw_output[0][0])

            last_name = current_name

            pbar.update()

        # Calculate the last 3D volume
        dice, asd = calculate_metrics(seg_output3D, y3D)
        dice_tot.append(dice)
        asd_tot.append(asd)
    print(sum(dice_tot)/len(dice_tot))
    print(sum(asd_tot)/len(asd_tot))



if __name__ == '__main__':
    pass
