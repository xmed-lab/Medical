#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 9:46
# @Author  : xxxx
# @Site    : 
# @File    : inference_fundus.py
# @Software: PyCharm

# #Desc:
import torch
import os
import cv2
import numpy as np
from config import config
from torch.utils.data import DataLoader
from load_fund import convert_labeled_list, OPTIC_dataset, collate_fn_wo_transform
from metrics_fund import calculate_metrics
from tqdm import tqdm


def save_images(model_path, loader, device, result_path):
    model = torch.load(model_path).to(device)
    print('model load sucess')

    dice_tot_d, dice_tot_c = [], []
    mask_type = torch.float32
    n_val = len(loader)
    model.eval() 
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, sample in enumerate(loader):
            imgs, y, path = sample['data'], sample['mask'], sample['name']

            imgs = imgs.to(device=device, dtype=torch.float32)
            y = y.to(dtype=mask_type)

            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= model(imgs, y, 'test')
                # reco, z_out, z_out_tilde, a_out, _, mu, logvar, cls_out, _ = net(imgs, true_masks, 'test')
            mask = mask[:,:2,:,:]
            seg_output = torch.sigmoid(mask.detach().cpu())
            dice_d, dice_c = calculate_metrics(seg_output, y)
            dice_tot_d.append(dice_d)
            dice_tot_c.append(dice_c)


            draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
            img_name = path[0].split('/')[-2] + '-' + path[0].split('/')[-1].split('.')[0]
            imgs = cv2.cvtColor((imgs[0].detach().cpu().numpy()*255).transpose(1,2,0), cv2.COLOR_BGR2RGB)

            # img & gt
            cv2.imwrite(result_path + '/' + img_name + '_img.png', imgs)
            # cv2.imwrite(result_path + '/' + img_name + '_gt_oc.png', (y[0][1].detach().cpu().numpy() * 255).astype(np.uint8))
            # cv2.imwrite(result_path + '/' + img_name + '_gt_od.png', (y[0][0].detach().cpu().numpy() * 255).astype(np.uint8))
            #
            # cv2.imwrite(result_path + '/' + img_name + '_pred_oc.png', draw_output[0][1])
            # cv2.imwrite(result_path + '/' + img_name + '_pred_od.png', draw_output[0][0])

            # # Optic Cup
            # cv2.imwrite(result_path + '/' + img_name + '_OC.png', draw_output[0][1])
            # # Optic Disc
            # cv2.imwrite(result_path + '/' + img_name + '_OD.png', draw_output[0][0])
            pbar.update()


        dice_mean_d, dice_mean_c = np.mean(dice_tot_d), np.mean(dice_tot_c)
        print(dice_mean_d, dice_mean_c)




if __name__ == '__main__':
    root = '/home/czhaobo/Medical/log/ckps'
    model_name = 'promptransunet'
    dataset = 'FUN'
    batch_sz = 1
    test_vendor = 'Drishti_GS'# 'REFUGE'
    ratio = '1'  # 0.02
    pth = 'CP_29.pth'
    aba = 'lora'
    model_path = os.path.join(root, model_name + '-' + dataset + '-' + test_vendor + '-' + ratio + aba, pth)
    img_path = '/home/czhaobo/Medical/draw/fund'
    device = 'cuda:4'


    target_test_csv = [test_vendor + '_test.csv', test_vendor + '_train.csv']
    ts_img_list, ts_label_list = convert_labeled_list(config['ROOT'], target_test_csv)

    target_valid_dataset = OPTIC_dataset(config['ROOT'], ts_img_list, ts_label_list,
                                         config['CROP_SIZE'], img_normalize=True)


    test_loader = DataLoader(dataset=target_valid_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             collate_fn=collate_fn_wo_transform,
                             num_workers=0)
    print('data load success~')
    save_images(model_path, test_loader,  device, img_path)


