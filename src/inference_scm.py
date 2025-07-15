#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 20:09
# @Author  : xxx
# @Site    : 
# @File    : inference_scm.py
# @Software: PyCharm

# #Desc: # 利用训练好的checkpoint进行inference
import torch
import os
import models
import torch.nn as nn
from functools import partial
from config import config
# from main_scm import prepare_data
from loader_scm import prepare_inference_loader
from utils import im_convert, save_image
import torch.nn.functional as F
import statistics
from metrics import dice_coeff,hausdorff_distance




def inference(model_path,batch_sz, test_vendor):
    # models
    # # load model
    model = torch.load(model_path)
    device = torch.device('cuda:'+str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    model = model.to(device)
    print('model load success')

    device = torch.device('cuda:'+str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    model = model.to(device)
    # load data
    datas = prepare_inference_loader(batch_sz, test_vendor)
    test_loader = datas
    model.eval()
    flag='000'
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
            score = sum(tot_sub) / len(tot_sub)  # for one patient
            score_hsd = sum(tot_sub_hsd) / len(tot_sub_hsd)
            tot.append(score)
            tot_hsd.append(score_hsd)


            tot_sub = []
            tot_sub_hsd = []


            if score <= 0.7:
                record_flag[flag] = score
            flag = path_img[0][-10: -7]

        with torch.no_grad():
            mask, reco, z_out, z_out_tilde, cls_out, fea_sets ,_,_,_= model(imgs, None, 'test')
        mask = mask[:, :2, :, :]
        pred = F.softmax(mask, dim=1)
        pred = (pred > 0.5).float()
        dice = dice_coeff(pred[:, 0, :, :], true_mask[:, 0, :, :], device).item()
        hsd = hausdorff_distance(pred[:, 0, :, :], true_mask[:, 0, :, :])

        tot_sub.append(dice)
        tot_sub_hsd.append(hsd)


    tot.append(sum(tot_sub) / len(tot_sub))
    tot_hsd.append(sum(tot_sub_hsd) / len(tot_sub_hsd))


    for i in range(len(tot)):
        tot[i] = tot[i] * 100
        tot_hsd[i] = tot_hsd[i] * 100

    print("Test Num: ", len(tot))
    print("Dice list: ",tot)
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
        mask, reco, z_out, z_out_tilde, cls_out, fea_sets,_,_,_= model(imgs, None, 'test')
    mask = mask[:, :2, :, :]

    pred = F.softmax(mask, dim=1)
    pred = (pred > 0.5).float()
    dice = dice_coeff(pred[:, 0, :, :], true_mask[:, 0, :, :], device).item()

    image = imgs
    # torch.set_printoptions(threshold=np.inf)
    # with open('./test.txt', 'wt') as f:
    #     print(onehot_predmax==mask, file=f)
    pred = pred[:, 0, :, :]
    real_mask = mask[:, 0, :, :]

    print(img_path[0])
    print("dice score: ", dice)
    real_mask = im_convert(real_mask, False)
    image = im_convert(image, False)
    pred = im_convert(pred, False)
    save_image(real_mask, '../draw/img/scgm_gt' + str(test_vendor) + '.pdf')
    save_image(image, '../draw/img/scgm_image' + str(test_vendor) + '.pdf')
    save_image(pred, '../draw/img/scgm_pred' + str(test_vendor) + '.pdf')



def draw_many_img(model_path, batch_sz, test_vendor):
    model = torch.load(model_path)
    device = torch.device('cuda:' + str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    model = model.to(device)

    # load data
    datas = prepare_inference_loader(batch_sz, test_vendor)
    test_loader = datas
    model.eval()

    flag = '047'
    tot = 0
    tot_sub = []

    for imgs, masks, path_img in test_loader:

        imgs = imgs.to(device)
        true_mask = masks.to(device)
        if path_img[0][-10: -7] == flag:
            image_slice = path_img[0][-7:-4]
            with torch.no_grad():
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,_,_ = model(imgs, None, 'test')
            mask = mask[:, :2, :, :]

            pred = F.softmax(mask, dim=1)
            pred = (pred > 0.5).float()
            dice = dice_coeff(pred[:, 0, :, :], true_mask[:, 0, :, :], device).item()

            save_once(imgs, pred, mask, flag, image_slice)

            tot_sub.append(dice)
        else:
            pass

    print('dice is ', sum(tot_sub) / len(tot_sub))


def save_once(image, pred, mask, flag, image_slice):
    pred = pred[:, 0:3, :, :]
    real_mask = mask[:, 0:3, :, :]
    mask = im_convert(real_mask, True)
    image = im_convert(image, False)
    pred = im_convert(pred, True)

    save_image(mask, './pic/' + str(flag) + '/real_mask' + str(image_slice) + '.png')
    save_image(image, './pic/' + str(flag) + '/image' + str(image_slice) + '.png')
    save_image(pred, './pic/' + str(flag) + '/pred' + str(image_slice) + '.png')



if __name__ == '__main__':
    root = '/home/czhaobo/Medical/log/ckps'
    model_name = 'promptransunet'
    dataset = 'SCM'
    batch_sz = 1
    test_vendor = 'A'
    ratio = '1' # 0.02
    pth = 'CP_69.pth' # 51
    aba = 'LORA'
    model_path = os.path.join(root, model_name + '-'+ dataset + '-' + test_vendor + '-' + ratio + aba, pth)
    inference(model_path, batch_sz, test_vendor)
    # draw_img(model_path, batch_sz, test_vendor)
