#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 0:36
# @Author  : xxxx
# @Site    : 
# @File    : main_pro.py
# @Software: PyCharm

# #Desc:



import os
import gc
import torch
import logging
import models
import utils
import losses
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
# 忽略 PyTorch 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
from config import config
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from load_fund import normalize_image,convert_labeled_list, OPTIC_dataset, collate_fn_wo_transform, collate_fn_w_transform
from torch.utils.tensorboard import SummaryWriter
from eval_fund import eval_dgnet
from metrics import FocalLoss, ContraLoss
from utils import KL_divergence

from torchnet import meter
import numpy as np
np.bool = np.bool_

from torch.cuda.amp import autocast as autocast

def latent_norm(a):
    """向量norm，除了batch和channel外"""
    n_batch, n_channel, _, _ = a.size()
    for batch in range(n_batch):
        for channel in range(n_channel):
            a_min = a[batch, channel, :, :].min()
            a_max = a[batch, channel, :, :].max()
            a[batch, channel, :, :] += a_min
            a[batch, channel, :, :] /= a_max - a_min
    return a



def prepare_data(cfg, batch_sz, test_vendor):
    """这里在预处理时候没有操作，所以需要shuffle"""
    ###
    source_name, target_name = list(set(['Drishti_GS', 'ORIGA', 'REFUGE', 'BinRushed','Magrabia']) - set([test_vendor])), test_vendor
    source_name = sorted(source_name)
    print(source_name, target_name)
    source_csv = [source_name[0] + '_test.csv', source_name[0] + '_train.csv']
    sr_img_list, sr_label_list = convert_labeled_list(cfg["ROOT"], source_csv)
    domain_1_labeled_dataset = OPTIC_dataset(cfg["ROOT"], sr_img_list, sr_label_list,
                                      cfg["CROP_SIZE"], img_normalize=False, reid=[1,0,0,0])


    source_csv = [source_name[1] + '_test.csv', source_name[1] + '_train.csv']
    sr_img_list, sr_label_list = convert_labeled_list(cfg["ROOT"], source_csv)
    domain_2_labeled_dataset = OPTIC_dataset(cfg["ROOT"], sr_img_list, sr_label_list,
                                      cfg["CROP_SIZE"], img_normalize=False, reid=[0,1,0,0])


    source_csv = [source_name[2] + '_test.csv', source_name[2] + '_train.csv']
    sr_img_list, sr_label_list = convert_labeled_list(cfg["ROOT"], source_csv)
    domain_3_labeled_dataset = OPTIC_dataset(cfg["ROOT"], sr_img_list, sr_label_list,
                                      cfg["CROP_SIZE"], img_normalize=False, reid=[0,0,1,0])


    source_csv = [source_name[3] + '_test.csv', source_name[3] + '_train.csv']
    sr_img_list, sr_label_list = convert_labeled_list(cfg["ROOT"], source_csv)
    domain_4_labeled_dataset = OPTIC_dataset(cfg["ROOT"], sr_img_list, sr_label_list,
                                      cfg["CROP_SIZE"], img_normalize=False, reid=[0,0,0,1])


    target_test_csv = [target_name + '_test.csv',target_name + '_train.csv']
    ts_img_list, ts_label_list = convert_labeled_list(config['ROOT'], target_test_csv)

    target_valid_dataset = OPTIC_dataset(config['ROOT'], ts_img_list, ts_label_list,
                                            config['CROP_SIZE'], img_normalize=True)
    test_loader = DataLoader(dataset=target_valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=False,
                              collate_fn=collate_fn_wo_transform,
                              num_workers=0)


    ###
    d_len = []
    d_len.append(len(domain_1_labeled_dataset))
    d_len.append(len(domain_2_labeled_dataset))
    d_len.append(len(domain_3_labeled_dataset))
    d_len.append(len(domain_4_labeled_dataset))


    # Label data balance 必须要balance到最大的，否则无法循环；
    long_len = max(d_len) # max dataset
    print('Max len of source domains ', long_len)
    print('Before balance: Domain 1,2,3,4,5: ', d_len)
    new_d_1 = domain_1_labeled_dataset
    for i in range(long_len // d_len[0]):
        if long_len == d_len[0]:
            break
        new_d_1 = ConcatDataset([new_d_1, domain_1_labeled_dataset])
    domain_1_labeled_dataset = new_d_1
    domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=0, pin_memory=False, collate_fn=collate_fn_w_transform)

    new_d_2 = domain_2_labeled_dataset
    for i in range(long_len // d_len[1]):
        if long_len == d_len[1]:
            break
        new_d_2 = ConcatDataset([new_d_2, domain_2_labeled_dataset])
    domain_2_labeled_dataset = new_d_2
    domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=0, pin_memory=False, collate_fn=collate_fn_w_transform)

    new_d_3 = domain_3_labeled_dataset
    for i in range(long_len // d_len[2]):
        if long_len == d_len[2]:
            break
        new_d_3 = ConcatDataset([new_d_3, domain_3_labeled_dataset])
    domain_3_labeled_dataset = new_d_3
    domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=0, pin_memory=False, collate_fn=collate_fn_w_transform) # 这个最好和imagefolder只留一个，不然有可能会造成死锁；

    new_d_4 = domain_4_labeled_dataset
    for i in range(long_len // d_len[3]):
        if long_len == d_len[3]:
            break
        new_d_4 = ConcatDataset([new_d_4, domain_4_labeled_dataset])
    domain_4_labeled_dataset = new_d_4
    domain_4_labeled_loader = DataLoader(dataset=domain_4_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=0, pin_memory=False, collate_fn=collate_fn_w_transform) # 这个最好和imagefolder只留一个，不然有可能会造成死锁；


    print("After balance Domain=1,2,3,4: ", len(domain_1_labeled_dataset), len(domain_2_labeled_dataset), len(domain_3_labeled_dataset), len(domain_4_labeled_dataset))
    print("Len of balanced labeled dataloader: Domain 1: {}, Domain 2: {}, Domain 3: {}, Domain 4: {}.".\
          format(len(domain_1_labeled_loader), len(domain_2_labeled_loader), len(domain_3_labeled_loader),  len(domain_4_labeled_dataset)))

    return long_len, domain_1_labeled_loader, domain_2_labeled_loader, domain_3_labeled_loader, domain_4_labeled_loader,test_loader


def train(config):
    # params
    best_score, best_1, best_2, best_3 = 0, 0, 0, 0 # np.inf
    loss_meter = meter.AverageValueMeter()

    epochs = config['EPOCH']
    batch_sz = config['BATCH']
    lr = config['LR']
    test_vendor = config['TASK']
    device = config['device']
    dir_checkpoint = os.path.join(config['LOG'], 'ckps', config['MODEL'] + '-' + config['DATA'] + '-' + config['TASK'] +'-' + str(config['RATIO'])+'lora')
    wc = os.path.join(config['LOG'], 'logdirs', config['MODEL'] + '-' + config['DATA'] + '-' + config['TASK'] +'-' +str(config['RATIO'])+ 'lora') # tensorboard
    if not os.path.exists(dir_checkpoint):
        print("create the ckpt dirs")
        os.makedirs(dir_checkpoint)
    if not os.path.exists(wc):
        print("create the log dirs")
        os.makedirs(wc)

    opt_patience = 4
    k1 = 1 # 原来是30
    k2 = 2 # 是其的倍数

    # models
    model_params = {}
    decoder_params = {
        'width': config['CROP_SIZE'],
        'prompt_dim': 256,
        'head': 8,
        'transformer_dim': 256,
        'anatomy_out_channels':2,
        'decoder_type': config['DECODER'],
        'z_length': 8,
        'num_mask_channels': 2} # 为啥要输出8个，因为有8个解剖学器官的先验，这里我们会要求前四个保持一致
    global_params = {
        "prompt_size": 8, # 感觉三十不影响才对
        "domain_num" : 4,}
    encoder_params = {
        "pretrained": True,
        "pretrained_vit_name": 'vit_base_patch16_224_in21k',
        "pretrained_folder" : r'/home/czhaobo/Medical/src/models',
        "img_size": config['CROP_SIZE'],
        "num_frames" : 8,
        "patch_size" : 16,
        "in_chans" : 3,
        "embed_dim" : 768,
        "depth" : 12,
        "num_heads" : 12,
        "mlp_ratio" : 4.,
        "patch_embedding_bias" : True,
        "qkv_bias" : True,
        "qk_scale" : None,
        "drop_rate" : 0.,
        "attn_drop_rate" : 0.,
        "drop_path_rate" : 0.2,
        "norm_layer" : partial(nn.LayerNorm, eps=1e-6),
        "adapt_method": False,
        "num_domains": 1,
        "prompt_num": 1,
    }
    model_params.update(decoder_params)
    model_params.update(global_params)
    model_params.update(encoder_params)


    model = models.get_model(config['MODEL'], model_params)
    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    print('Encoder Parameters: ', utils.count_parameters(model.vision_encoder.resnet)) # 冻结了vit的参数
    # print('Encoder Parameters: ', utils.count_parameters(model.vision_encoder.vit))
    # print('Decoder Parameters: ', utils.count_parameters(model.decoder), utils.count_parameters(model.decoder.seg_head), utils.count_parameters(model.decoder.recon_head))

    # models.initialize_weights(model, config['WEIGHT_INIT']) # 千万不能这么搞，艹
    model.to(device)
    # model = nn.DataParallel(model, device_ids=[0,1], output_device=0)


    # data set
    cfg = {}
    cfg['ROOT'] = config['ROOT']
    cfg['CROP_SIZE'] = config['CROP_SIZE']
    cfg['BATCH'] = config['BATCH']

    long_len, domain_1_labeled_loader,domain_2_labeled_loader,domain_3_labeled_loader,\
    domain_4_labeled_loader,test_loader = prepare_data(cfg, batch_sz, test_vendor)


    # metric & criterion
    # l2_distance = nn.MSELoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    l1_distance = nn.L1Loss().to(device)
    contra_loss = ContraLoss(config['TEMP'])
    bce_loss = torch.nn.BCELoss()

    focal = FocalLoss()

    # optimizer initialization
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    params_to_optimize = [
        {'params': model.global_prompt.parameters()},
        {'params': model.vision_encoder.parameters()},
        {'params': model.prompt_bank},
        {'params': model.decoder.parameters()}
    ]
    params_to_optimize2 = [
        {'params': model.adapter.parameters()},
    ]
    optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=5e-4) # 这玩意没有schedule不得注意点
    optimizer2 = optim.Adam(params_to_optimize2, lr=lr*0.1, weight_decay=5e-4) #

    writer = SummaryWriter(log_dir=wc)

    # train
    global_step = 0
    for epoch in range(epochs):
        # data preprocess
        model.train()
        with tqdm(total=long_len, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            domain_1_labeled_itr = iter(domain_1_labeled_loader) # iterator
            domain_2_labeled_itr = iter(domain_2_labeled_loader)
            domain_3_labeled_itr = iter(domain_3_labeled_loader)
            domain_4_labeled_itr = iter(domain_4_labeled_loader)

            domain_labeled_iter_list = [domain_1_labeled_itr, domain_2_labeled_itr, domain_3_labeled_itr,domain_4_labeled_itr]

            # num iter: iteration
            for num_itr in range(long_len//batch_sz): # 不用除2的原因是因为meta_test会复制两次
                # Randomly choosing meta train and meta test domains every iters; 其实这也就相当于meta-learning的support set和eval set
                domain_list = np.random.permutation(4)
                meta_train_domain_list = domain_list[:2]
                meta_test_domain_list = domain_list[3]

                # 下面只用三个目前
                meta_train_imgs, meta_train_masks, meta_train_labels = [], [], []
                meta_test_imgs, meta_test_masks, meta_test_labels = [], [], []
                sample = next(domain_labeled_iter_list[meta_train_domain_list[0]])
                imgs, true_masks, labels = sample['data'], sample['mask'], sample['reid'].squeeze()
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)
                sample = next(domain_labeled_iter_list[meta_train_domain_list[1]])
                imgs, true_masks, labels = sample['data'], sample['mask'], sample['reid'].squeeze()
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)

                sample = next(domain_labeled_iter_list[meta_test_domain_list]) # why twice
                imgs, true_masks, labels = sample['data'], sample['mask'], sample['reid'].squeeze()
                meta_test_imgs.append(imgs) # why twice;因为要匹配两个域,而且因为labeled的dataset的batch_size//2了
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)
                sample = next(domain_labeled_iter_list[meta_test_domain_list])
                imgs, true_masks, labels = sample['data'], sample['mask'], sample['reid'].squeeze()
                meta_test_imgs.append(imgs)
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)


                meta_train_imgs = torch.cat((meta_train_imgs[0], meta_train_imgs[1]), dim=0) # 两个meta-source域
                meta_train_masks = torch.cat((meta_train_masks[0], meta_train_masks[1]), dim=0)
                meta_train_labels = torch.cat((meta_train_labels[0], meta_train_labels[1]), dim=0)
                meta_test_imgs = torch.cat((meta_test_imgs[0], meta_test_imgs[1]), dim=0) # meta_test这里其实也对应两个
                meta_test_masks = torch.cat((meta_test_masks[0], meta_test_masks[1]), dim=0)
                meta_test_labels = torch.cat((meta_test_labels[0], meta_test_labels[1]), dim=0)

                # test_score = eval_dgnet(model, test_loader, device, mode='test')  # 这里才是测试得分

                # train preprocess
                # supervise loss
                total_meta_loss = 0.0
                # meta-train: 1. load meta-train data 2. calculate meta-train loss
                ###############################Meta train#######################################################
                imgs = meta_train_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = meta_train_masks.to(device=device, dtype=mask_type)
                labels = meta_train_labels.to(device=device, dtype=torch.float32)

                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar ,domain_cls_1= model(imgs, labels, 'training')

                kl_loss1 = KL_divergence(logvar[:, :8], mu[:, :8])
                kl_loss2 = KL_divergence(logvar[:, 8:], mu[:, 8:])
                prior_loss = (kl_loss1 + kl_loss2) * config['prior_reg']


                ssl_loss = contra_loss(fea_sets)

                seg_pred = mask[:, :2, :, :] # 选前2个

                reco_loss = 0#l1_distance(reco, imgs)
                regression_loss = 0 # l1_distance(z_out_tilde, z_out)

                sf_seg = F.sigmoid(seg_pred)
                dice_loss_1 = losses.dice_loss_pro(sf_seg[:, 0,:,:], true_masks[:,0,:,:]) + bce_loss(sf_seg[:, 0,:,:], true_masks[:,0,:,:])
                dice_loss_2 = losses.dice_loss_pro(sf_seg[:, 1,:,:], true_masks[:,1,:,:]) + bce_loss(sf_seg[:, 1,:,:], true_masks[:,1,:,:])

                loss_dice = dice_loss_1 + 1.5*dice_loss_2 # 别的都是2， OC

                loss_focal = 0 # focal(seg_pred_swap, ce_target) # 这里为啥要使用focal loss，因为不同的目标的大小不一样？
                d_cls = criterion(cls_out, labels) + criterion(domain_cls_1, labels)
                d_losses = d_cls * config['DLS']

                batch_loss = prior_loss  +reco_loss+ (config['RECON'] * regression_loss) + 5*loss_dice + 5*loss_focal + d_losses + config["SSL"] * ssl_loss
                # print("A ",5*loss_dice,5*loss_focal, reco_loss, d_losses ,config['SSL'] * ssl_loss , config['RECON']*regression_loss)

                total_meta_loss += batch_loss

                #
                #writer.add_scalar('Meta_train_Loss/loss_dice', loss_dice.item(), global_step)
                #writer.add_scalar('Meta_train_Loss/dice_loss_1', dice_loss_1.item(), global_step)
                #writer.add_scalar('Meta_train_Loss/dice_loss_2', dice_loss_2.item(), global_step)
                #writer.add_scalar('Meta_train_Loss/loss_d_cls', d_cls.item(), global_step)
                #writer.add_scalar('Meta_train_Loss/loss_ssl', ssl_loss.item(), global_step)
                #writer.add_scalar('Meta_train_Loss/batch_loss', batch_loss.item(), global_step)


                # meta-test: 1. load meta-test data 2. calculate meta-test loss
                ###############################Meta test#######################################################
                imgs = meta_test_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = meta_test_masks.to(device=device, dtype=mask_type)
                labels = meta_test_labels.to(device=device, dtype=torch.float32)
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar, domain_cls_1 = model(imgs, labels, 'training', meta_loss=None
                                                                          ) # batch_loss

                ssl_loss = contra_loss(fea_sets)

                reco_loss = 0 # l1_distance(reco, imgs)
                regression_loss = 0# l1_distance(z_out_tilde, z_out)

                seg_pred = mask[:, :2, :, :]
                sf_seg  = F.sigmoid(seg_pred)
                dice_loss_1 = losses.dice_loss_pro(sf_seg[:,0,:,:], true_masks[:,0,:,:]) + bce_loss(sf_seg[:, 0,:,:], true_masks[:,0,:,:])
                dice_loss_2 = losses.dice_loss_pro(sf_seg[:,1,:,:], true_masks[:,1,:,:]) + bce_loss(sf_seg[:, 1,:,:], true_masks[:,1,:,:])
                loss_dice = dice_loss_1 + 1.5* dice_loss_2 # + dice_loss_2 + dice_loss_3

                loss_focal = 0 #  focal(seg_pred_swap, ce_target) # 前面是background，不需要似乎。

                d_cls = criterion(cls_out, labels) + criterion(domain_cls_1, labels)
                d_losses = d_cls * config['DLS']

                batch_loss = 5*loss_dice + 5*loss_focal  +reco_loss+ d_losses + config['SSL'] * ssl_loss + config['RECON']*regression_loss
                # print("B ",5*loss_dice,5*loss_focal, reco_loss, d_losses ,config['SSL'] * ssl_loss , config['RECON']*regression_loss)

                total_meta_loss += batch_loss

                optimizer.zero_grad()
                optimizer2.zero_grad()
                total_meta_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                optimizer2.step()

                loss_meter.add(total_meta_loss.sum().item())

                pbar.set_postfix(**{'loss (batch)': total_meta_loss.item()})
                pbar.update(imgs.shape[0])
                global_step += 1

            if optimizer.param_groups[0]['lr']<=2e-8:
                print('Converge')

            # if best_dice_loss > loss_meter.value()[0]:
            #     best_dice_loss = loss_meter.value()[0]
            #     try:
            #         os.mkdir(dir_checkpoint)
            #         logging.info('Created checkpoint directory')
            #     except OSError:
            #         pass
            #     # torch.save(model,
            #     #            dir_checkpoint + '/CP_{}.pth'.format(epoch))
            #     logging.info('Checkpoint saved !') # 艹，这种方式会导致模型学习的最优点可能被省略

            if (epoch + 1) > k1 and (epoch + 1) % k2 == 0:
                model.eval()
                # 测试
                test_score_d, test_score_c = eval_dgnet(model, test_loader, device, mode='test') # 这里才是测试得分
                print("Current epoch(Dice),", test_score_d, test_score_c)
                #writer.add_scalar('Dice/test_d', test_score_d, epoch) # 比较高
                #writer.add_scalar('Dice/test_c', test_score_c, epoch)

                if best_score < test_score_c:  # 只汇报目前最好情况
                    best_score = test_score_c
                    best_1 = test_score_d

                    print("Epoch checkpoint")
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(model, dir_checkpoint + '/CP_{}.pth'.format(epoch))
                # logging.info('Best Dice Coeff: {}'.format(best_dice))
                print(best_score, best_1)


            else:
                pass


            gc.collect()
            torch.cuda.empty_cache()

        writer.close()




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    config = config
    device = torch.device('cuda:'+str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    config['device'] = device
    logging.info(f'Using device {device}')

    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)

    train(config)


