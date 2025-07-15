#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 12:23
# @Author  : xxxx
# @Site    : 
# @File    : inference_others.py
# @Software: PyCharm

# #Desc: # 这里设置main_others
import torch
import os
from tqdm import tqdm
from metrics import dice_coeff
from load_others import get_seg_dg_dataloader
from config import config
import numpy as np
import cv2
from torchmetrics import Accuracy, AUROC, Specificity
from matplotlib import pyplot as plt

def save_images(model_path, loader, device, result_path):
    model = torch.load(model_path).to(device)
    print('model load sucess')
    model.eval()  # 没有model.eval直接会不一致；


    # auroc = AUROC().cuda()
    # accuracy = Accuracy().cuda()
    auroc = Accuracy(task="binary", num_classes=2).to(device)
    accuracy = AUROC(task="binary", num_classes=2).to(device)

    tot = 0
    tot_auc, tot_acc = 0,0
    tots,tot_aucs, tot_accs,names = [],[],[],[]
    mask_type = torch.float32
    n_val = len(loader)


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, sample in enumerate(loader):
            imgs, true_masks = sample['image'], sample['label']
            path = sample['img_name']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= model(imgs, true_masks, 'test')

            mask_pred = mask[:, :1, :, :]

            # seg_pred = torch.sigmoid(mask_pred)
            preds = torch.sigmoid(mask_pred)
            pred = (preds > 0.5).float()
            # preds = preds[:,0,:,:]
            dsc = dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
            # print(torch.stack([1 - preds, preds], dim=1).shape)
            # print(true_masks[:,0,:,:].long().shape)
            acc = accuracy(preds[:, 0, :, :], true_masks[:,0,:,:].long())
            aucroc = auroc(preds[:, 0, :, :], true_masks[:,0,:,:].long())

            # dsc = f1_score(torch.stack([1 - seg_pred[:, 0], seg_pred[:, 0]], dim=1), true_masks[:, 0].long())[1]
            tot += dsc
            tot_acc += acc
            tot_auc += aucroc
            tots.append(dsc)
            tot_accs.append(acc)
            tot_aucs.append(aucroc)
            names.append(path)

            # img draw
            draw_output = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
            imgs = cv2.cvtColor((imgs[0].detach().cpu().numpy()*255).transpose(1,2,0), cv2.COLOR_BGR2RGB)

            # print(draw_output.shape, imgs.shape, true_masks.shape)
            # print(path)
            img_name = path[0].split('.')[0]

            # cv2.imwrite(result_path + '/' + img_name + '_img.png', imgs)
            # cv2.imwrite(result_path + '/' + img_name + '_gt.png', (true_masks[0][0].detach().cpu().numpy() * 255).astype(np.uint8))
            # cv2.imwrite(result_path + '/' + img_name + '_pred.png', draw_output[0][0])
            # Save input image as PDF
            plt.imsave(os.path.join(result_path, f"{img_name}_img.pdf"), imgs,
                       format='pdf')

            # Save ground truth mask as PDF (scale to [0, 255] and convert to uint8)
            plt.imsave(os.path.join(result_path, f"{img_name}_gt.pdf"),
                       (true_masks[0][0].detach().cpu().numpy() * 255).astype(np.uint8), format='pdf', cmap='gray')

            # Save predicted mask as PDF (directly from draw_output)
            plt.imsave(os.path.join(result_path, f"{img_name}_pred.pdf"), draw_output[0][0], format='pdf', cmap='gray')
            #
            pbar.update()

        print(names)
        print("Dice",tots)
        print("ACC", tot_accs)
        print("AUC", tot_aucs)
        print(tot/n_val)
        print(tot_auc/n_val)
        print(tot_acc/n_val)



if __name__ == '__main__':
    root = '/home/xxx/Medical/log/ckps'
    model_name = 'promptransunet'
    dataset = 'RVS'
    batch_sz = 1
    test_vendor = '3' # 1
    ratio = '1'
    pth = 'CP_111.pth' # 97
    aba = 'lora'
    model_path = os.path.join(root, model_name + '-' + dataset + '-' + test_vendor + '-' + ratio + aba, pth)
    img_path = '/home/czhaobo/Medical/draw/ret2'
    device = 'cuda:4'
    
    cfg = {}
    cfg['ROOT'] = config['ROOT']
    cfg['DATASET'] ='rvs'
    cfg['TRAIN'] = [0]
    cfg['TEST'] = [int(test_vendor)]

    _,_, test_loader = get_seg_dg_dataloader(cfg, batch_sz, workers=1, reid=3) # 有没有不打紧reid
    print("data load sucess!")


    save_images(model_path, test_loader, device, img_path)




