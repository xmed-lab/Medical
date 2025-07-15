#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 20:09
# @Author  : Anonymous
# @Site    :
# @File    : inference.py
# @Software: PyCharm

# #Desc: # inference
import torch
import os
import statistics
from config import config
from utils import im_convert, save_image
from metrics import dice_coeff,hausdorff_distance
import torch.nn.functional as F

from loader import prepare_inference_loader



def inference(model_path,batch_sz, test_vendor):
    # # load model
    model = torch.load(model_path)
    device = torch.device('cuda:'+str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    model = model.to(device)
    # load data
    datas = prepare_inference_loader(batch_sz, test_vendor)
    test_loader = datas
    model.eval()

    tot = []
    tot_hsd = []

    tot_sub = []
    tot_sub_hsd = []

    flag = '000'
    record_flag = {}
    for imgs, masks, path_img in test_loader:

        imgs = imgs.to(device)
        true_mask = masks.to(device)

        if path_img[0][-10: -7] != flag:
            score = sum(tot_sub) / len(tot_sub) # for one patient
            score_hsd = sum(tot_sub_hsd) / len(tot_sub_hsd)
            tot.append(score)
            tot_hsd.append(score_hsd)

            tot_sub = []
            tot_sub_hsd = []

            if score <= 0.7:
                record_flag[flag] = score
            flag = path_img[0][-10: -7]

        with torch.no_grad():
            mask, reco, z_out, z_out_tilde, cls_out, fea_sets,_,_,_ = model(imgs, None, 'test')
        mask = mask[:, :4, :, :] 
        pred = F.softmax(mask, dim=1)
        pred = (pred > 0.5).float()
        dice = dice_coeff(pred[:, 0:3, :, :], true_mask[:, 0:3, :, :], device).item()
        hsd = hausdorff_distance(pred[:, 0:3, :, :], true_mask[:, 0:3, :, :])

        tot_sub.append(dice)
        tot_sub_hsd.append(hsd)

    tot.append(sum(tot_sub) / len(tot_sub))
    tot_hsd.append(sum(tot_sub_hsd) / len(tot_sub_hsd))

    for i in range(len(tot)):
        tot[i] = tot[i] * 100

    print("Test Num: ", len(tot))
    print("Dice list: ", tot)
    print("Dice mean: ", sum(tot) / len(tot))
    print("Dice std: ", statistics.stdev(tot))
    print("HD mean: ", sum(tot_hsd) / len(tot_hsd))
    print("HD std: ", statistics.stdev(tot_hsd))
    print("Weak sample: ", record_flag)

def draw_img(model_path, batch_sz, test_vendor):
    model = torch.load(model_path)
    device = torch.device('cuda:' + str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    model = model.to(device)

    # load data
    datas = prepare_inference_loader(batch_sz, test_vendor)
    test_loader = datas
    model.eval()

    dataiter = iter(test_loader)
    minibatch = next(dataiter)
    imgs = minibatch[0].to(device)
    true_mask = minibatch[1].to(device)
    img_path = minibatch[2]

    with torch.no_grad():
        mask, reco, z_out, z_out_tilde, cls_out, fea_sets,_,_,_ = model(imgs, None, 'test')
    mask = mask[:,:4,:,:]
    pred = F.softmax(mask, dim=1)
    pred = (pred > 0.5).float()
    dice = dice_coeff(pred[:, 0:3, :, :], true_mask[:, 0:3, :, :], device).item()

    image = imgs
    # torch.set_printoptions(threshold=np.inf)
    # with open('./test.txt', 'wt') as f:
    #     print(onehot_predmax==mask, file=f)
    pred = pred[:, 0:3, :, :]
    real_mask = true_mask[:, 0:3, :, :]

    print(img_path[0])
    print("dice score: ", dice)
    real_mask = im_convert(real_mask, False)
    image = im_convert(image, False)
    pred = im_convert(pred, False)
    save_image(real_mask, '../draw/gpl/mm_gt' + str(test_vendor) + '.png')
    save_image(image, '../draw/gpl/mm_image' + str(test_vendor) + '.png')
    save_image(pred, '../draw/gpl/mm_pred' + str(test_vendor) + '.png')






if __name__ == '__main__':
    root = '/home/xxxx/Medical/log/ckps'
    model_name = 'promptransunet'
    dataset = 'MM'
    batch_sz = 1
    test_vendor = 'D'
    ratio = '0.02' # 0.02
    pth = 'XX.pth'

    model_path = os.path.join(root, model_name + '-'+ dataset + '-' + test_vendor + '-' + ratio, pth)
    inference(model_path, batch_sz, test_vendor)
