#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:36
# @Author  : Anonymous
# @Site    :
# @File    : main.py
# @Software: PyCharm

# #Desc: for mms
import gc
import os
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
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
from config import config
from functools import partial
from torch.utils.data import DataLoader, random_split, ConcatDataset
from loader import get_meta_split_data_loaders
from torch.utils.tensorboard import SummaryWriter
from eval import eval_dgnet
from metrics import FocalLoss, ContraLoss
from utils import KL_divergence




def latent_norm(a):
    n_batch, n_channel, _, _ = a.size()
    for batch in range(n_batch):
        for channel in range(n_channel):
            a_min = a[batch, channel, :, :].min()
            a_max = a[batch, channel, :, :].max()
            a[batch, channel, :, :] += a_min
            a[batch, channel, :, :] /= a_max - a_min
    return a





def prepare_data(batch_sz, test_vendor):
    _, domain_1_unlabeled_dataset, \
    _, domain_2_unlabeled_dataset, \
    _, domain_3_unlabeled_dataset, \
    test_loader, \
    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset = get_meta_split_data_loaders(
        batch_sz // 2, test_vendor=test_vendor, image_size=config['CROP_SIZE'])  

    val_dataset = ConcatDataset(
        [domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset]) 

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_sz, shuffle=False, drop_last=True, pin_memory=False,
                            num_workers=2) 

    d_len = []
    d_len.append(len(domain_1_labeled_dataset))
    d_len.append(len(domain_2_labeled_dataset))
    d_len.append(len(domain_3_labeled_dataset))


    # Label data balance 
    long_len = max(d_len) # max dataset
    print('Max len of source domains ', long_len)
    print('Before balance: Domain 1,2,3: ', d_len)
    new_d_1 = domain_1_labeled_dataset
    for i in range(long_len // d_len[0]):
        if long_len == d_len[0]:
            break
        new_d_1 = ConcatDataset([new_d_1, domain_1_labeled_dataset])
    domain_1_labeled_dataset = new_d_1
    domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_sz // 2, shuffle=False,
                                         drop_last=True, num_workers=2, pin_memory=False)

    new_d_2 = domain_2_labeled_dataset
    for i in range(long_len // d_len[1]):
        if long_len == d_len[1]:
            break
        new_d_2 = ConcatDataset([new_d_2, domain_2_labeled_dataset])
    domain_2_labeled_dataset = new_d_2
    domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_sz // 2, shuffle=False,
                                         drop_last=True, num_workers=2, pin_memory=False)

    new_d_3 = domain_3_labeled_dataset
    for i in range(long_len // d_len[2]):
        if long_len == d_len[2]:
            break
        new_d_3 = ConcatDataset([new_d_3, domain_3_labeled_dataset])
    domain_3_labeled_dataset = new_d_3
    domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_sz // 2, shuffle=False,
                                         drop_last=True, num_workers=2, pin_memory=False) 

    print("After balance Domain=1,2,3: ", len(domain_1_labeled_dataset), len(domain_2_labeled_dataset), len(domain_3_labeled_dataset))
    print("Len of balanced labeled dataloader: Domain 1: {}, Domain 2: {}, Domain 3: {}.".\
          format(len(domain_1_labeled_loader), len(domain_2_labeled_loader), len(domain_3_labeled_loader)))


    # Unlabel data
    d_len = []
    d_len.append(len(domain_1_unlabeled_dataset))
    d_len.append(len(domain_2_unlabeled_dataset))
    d_len.append(len(domain_3_unlabeled_dataset))
    print('Before balance: Domain 1,2,3: ', d_len)
    un_long_len = long_len * 1 

    new_d_1 = domain_1_unlabeled_dataset
    for i in range(un_long_len // d_len[0]):
        if un_long_len == d_len[0]:
            break
        new_d_1 = ConcatDataset([new_d_1, domain_1_unlabeled_dataset])
    domain_1_unlabeled_dataset = new_d_1
    domain_1_unlabeled_loader = DataLoader(dataset=domain_1_unlabeled_dataset, batch_size=batch_sz, shuffle=False,
                                         drop_last=True, num_workers=2, pin_memory=False)

    new_d_2 = domain_2_unlabeled_dataset
    for i in range(un_long_len // d_len[1]):
        if un_long_len == d_len[1]:
            break
        new_d_2 = ConcatDataset([new_d_2, domain_2_unlabeled_dataset])
    domain_2_unlabeled_dataset = new_d_2
    domain_2_unlabeled_loader = DataLoader(dataset=domain_2_unlabeled_dataset, batch_size=batch_sz, shuffle=False,
                                         drop_last=True, num_workers=2, pin_memory=False)

    new_d_3 = domain_3_unlabeled_dataset
    for i in range(un_long_len // d_len[2]):
        if un_long_len == d_len[2]:
            break
        new_d_3 = ConcatDataset([new_d_3, domain_3_unlabeled_dataset])
    domain_3_unlabeled_dataset = new_d_3
    domain_3_unlabeled_loader = DataLoader(dataset=domain_3_unlabeled_dataset, batch_size=batch_sz, shuffle=False,
                                         drop_last=True, num_workers=2,
                                         pin_memory=False)  

    print("After balance Domain=1,2,3: ", len(domain_1_unlabeled_dataset), len(domain_2_unlabeled_dataset), len(domain_3_unlabeled_dataset))
    print("Len of balanced unlabeled dataloader: Domain 1: {}, Domain 2: {}, Domain 3: {}.".\
          format(len(domain_1_unlabeled_loader), len(domain_2_unlabeled_loader), len(domain_3_unlabeled_loader)))



    return long_len, domain_1_labeled_loader,domain_2_labeled_loader,domain_3_labeled_loader, \
    domain_1_unlabeled_loader, domain_2_unlabeled_loader,domain_3_unlabeled_loader,\
        val_loader, test_loader




def train(config):
    # params
    best_dice, best_lv, best_myo, best_rv = 0, 0, 0, 0
    epochs = config['EPOCH']
    batch_sz = config['BATCH']
    lr = config['LR']
    test_vendor = config['TASK']
    device = config['device']
    dir_checkpoint = os.path.join(config['LOG'], 'ckps',
                                  config['MODEL'] + '-' + config['DATA'] + '-' + config['TASK'] + '-' + str(config['RATIO']))
    wc = os.path.join(config['LOG'], 'logdirs',
                      config['MODEL'] + '-' + config['DATA'] + '-' + config['TASK'] + '-' + str(config[
                          'RATIO']))  # tensorboard
    if not os.path.exists(wc):
        print("create the log dirs")
        os.makedirs(wc)

    opt_patience = 4
    k_un = 1 
    k1 = 1 
    k2 = 2 

    # models
    model_params = {}
    decoder_params = {
        'width': config['CROP_SIZE'],
        'prompt_dim': 256,
        'head': 8,
        'transformer_dim': 256,
        'anatomy_out_channels':12, 
        'decoder_type': config['DECODER'],
        'z_length': 8,
        'num_mask_channels': 12} 
    global_params = {
        "prompt_size":30,
        "domain_num" :3,}
    encoder_params = {
        "pretrained": True,
        "pretrained_vit_name": 'vit_base_patch16_224_in21k',
        "pretrained_folder" : r'/home/xxx/Medical/src/models',
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
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0,1], output_device=0)


    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    model.to(device)


    # data set
    long_len, domain_1_labeled_loader,domain_2_labeled_loader,domain_3_labeled_loader, \
    domain_1_unlabeled_loader, domain_2_unlabeled_loader,domain_3_unlabeled_loader,\
        val_loader, test_loader= prepare_data(batch_sz, test_vendor)


    # metric & criterion
    criterion = nn.CrossEntropyLoss().to(device)
    l1_distance = nn.L1Loss().to(device)
    contra_loss = ContraLoss(config['TEMP'])

    focal = FocalLoss()
 
    params_to_optimize = [
        {'params': model.global_prompt.parameters()},
        {'params': model.vision_encoder.parameters()},
        {'params': model.prompt_bank},
        {'params': model.decoder.parameters()}
    ]
    params_to_optimize2 = [
        {'params': model.adapter.parameters()},

    ]
    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-1)
    optimizer2 = optim.AdamW(params_to_optimize2, lr=lr*0.1, weight_decay=1e-1) 

    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5,
                                                     patience=opt_patience)  
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'max', factor=0.5,
                                                      patience=opt_patience)  

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
            domain_labeled_iter_list = [domain_1_labeled_itr, domain_2_labeled_itr, domain_3_labeled_itr]

            domain_1_unlabeled_itr = iter(domain_1_unlabeled_loader)
            domain_2_unlabeled_itr = iter(domain_2_unlabeled_loader)
            domain_3_unlabeled_itr = iter(domain_3_unlabeled_loader)
            domain_unlabeled_iter_list = [domain_1_unlabeled_itr, domain_2_unlabeled_itr, domain_3_unlabeled_itr]

            # num iter: iteration
            for num_itr in range(long_len//batch_sz):
                # Randomly choosing meta train and meta test domains every iters
                domain_list = np.random.permutation(3)
                meta_train_domain_list = domain_list[:2]
                meta_test_domain_list = domain_list[2]

                meta_train_imgs, meta_train_masks, meta_train_labels = [], [], []
                meta_test_imgs, meta_test_masks, meta_test_labels = [], [], []
                meta_test_un_imgs, meta_test_un_labels = [], []

                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_train_domain_list[0]])
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)

                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_train_domain_list[1]])
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)

                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_test_domain_list])
                meta_test_imgs.append(imgs) 
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)
                imgs, true_masks, labels = next(domain_labeled_iter_list[meta_test_domain_list])
                meta_test_imgs.append(imgs)
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)

                imgs, labels = next(domain_unlabeled_iter_list[meta_test_domain_list]) # why twice
                meta_test_un_imgs.append(imgs)
                meta_test_un_labels.append(labels)
                imgs, labels = next(domain_unlabeled_iter_list[meta_test_domain_list])
                meta_test_un_imgs.append(imgs)
                meta_test_un_labels.append(labels)

                meta_train_imgs = torch.cat((meta_train_imgs[0], meta_train_imgs[1]), dim=0)
                meta_train_masks = torch.cat((meta_train_masks[0], meta_train_masks[1]), dim=0)
                meta_train_labels = torch.cat((meta_train_labels[0], meta_train_labels[1]), dim=0)
                meta_test_imgs = torch.cat((meta_test_imgs[0], meta_test_imgs[1]), dim=0) 
                meta_test_masks = torch.cat((meta_test_masks[0], meta_test_masks[1]), dim=0)
                meta_test_labels = torch.cat((meta_test_labels[0], meta_test_labels[1]), dim=0)
                meta_test_un_imgs = torch.cat((meta_test_un_imgs[0], meta_test_un_imgs[1]), dim=0)
                meta_test_un_labels = torch.cat((meta_test_un_labels[0], meta_test_un_labels[1]), dim=0)


                meta_train_un_imgs, meta_train_un_labels = [], [] 
                for i in range(k_un): # one batch for 2 domains; data loader
                    train_un_imgs, train_un_labels = [], []
                    un_imgs, un_labels = next(domain_unlabeled_iter_list[meta_train_domain_list[0]]) # B/2,1,288,288 
                    train_un_imgs.append(un_imgs)
                    train_un_labels.append(un_labels)
                    un_imgs, un_labels = next(domain_unlabeled_iter_list[meta_train_domain_list[1]]) # B/2,1,288,288
                    train_un_imgs.append(un_imgs)
                    train_un_labels.append(un_labels)
                    meta_train_un_imgs.append(torch.cat((train_un_imgs[0], train_un_imgs[1]), dim=0))
                    meta_train_un_labels.append(torch.cat((train_un_labels[0], train_un_labels[1]), dim=0))

                # train preprocess
                total_meta_un_loss = 0.0
                for i in range(k_un): 
                    # meta-train: 1. load meta-train data 2. calculate meta-train loss
                    ###############################Meta train#######################################################
                    un_imgs = meta_train_un_imgs[i].to(device=device, dtype=torch.float32)
                    un_labels = meta_train_un_labels[i].to(device=device, dtype=torch.float32)

                    mask, un_reco, un_z_out, un_z_tilde, un_cls_out, fea_sets,mu,logvar,domain_cls_1 = model(un_imgs, un_labels, 'training')

                    kl_loss1 = KL_divergence(logvar[:, :8], mu[:, :8])
                    kl_loss2 = KL_divergence(logvar[:, 8:], mu[:, 8:])
                    prior_loss = (kl_loss1 + kl_loss2) * config['prior_reg']

                    ssl_loss = contra_loss(fea_sets)

                    un_reco_loss = l1_distance(un_reco, un_imgs) 
                    un_regression_loss = l1_distance(un_z_tilde, un_z_out) 
                    
                    d_cls = criterion(un_cls_out, un_labels) + criterion(domain_cls_1, un_labels) 
                    un_batch_loss = prior_loss + un_reco_loss + (config['RECON'] * un_regression_loss) + config['DLS'] * d_cls + config['SSL'] * ssl_loss
                    total_meta_un_loss += un_batch_loss

                    # meta-test: 1. load meta-test data 2. calculate meta-test loss
                    ###############################Meta test#######################################################
                    un_imgs = meta_test_un_imgs.to(device=device, dtype=torch.float32)
                    un_labels = meta_test_un_labels.to(device=device, dtype=torch.float32)
                    mask, un_reco, un_z_out, un_z_tilde, un_cls_out, fea_sets,_,_,domain_cls_1 = model(
                        un_imgs, un_labels, 'training', meta_loss=un_batch_loss)


                    ssl_loss = contra_loss(fea_sets)

                    un_reco_loss = l1_distance(un_reco, un_imgs)
                    un_regression_loss = l1_distance(un_z_tilde, un_z_out)

                    d_cls = criterion(domain_cls_1, un_labels) + criterion(un_cls_out, un_labels)
                    un_batch_loss = un_reco_loss + un_regression_loss * config['RECON'] + d_cls * config['DLS'] + ssl_loss * config['SSL']
                    total_meta_un_loss += un_batch_loss

                    writer.add_scalar('Meta_test_loss/un_reco_loss', un_reco_loss.item(), global_step)
                    writer.add_scalar('Meta_test_loss/un_regression_loss', un_regression_loss.item(), global_step)
                    writer.add_scalar('Meta_test_loss/un_d_cls', d_cls.item(), global_step)
                    writer.add_scalar('Meta_test_loss/un_ssl_loss', ssl_loss.item(), global_step)
                    writer.add_scalar('Meta_test_loss/un_batch_loss', un_batch_loss.item(), global_step)

                    optimizer.zero_grad()
                    optimizer2.zero_grad()
                    total_meta_un_loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()
                    optimizer2.step()


                # supervise loss
                total_meta_loss = 0.0
                # meta-train: 1. load meta-train data 2. calculate meta-train loss
                ###############################Meta train#######################################################
                imgs = meta_train_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = meta_train_masks.clone().to(device=device, dtype=torch.long)
                true_masks = meta_train_masks.to(device=device, dtype=mask_type)
                labels = meta_train_labels.to(device=device, dtype=torch.float32)

                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,domain_cls_1= model(imgs, labels, 'training')
                kl_loss1 = KL_divergence(logvar[:, :8], mu[:, :8])
                kl_loss2 = KL_divergence(logvar[:, 8:], mu[:, 8:])
                prior_loss = (kl_loss1 + kl_loss2) * config['prior_reg']

                ssl_loss = contra_loss(fea_sets)

                seg_pred = mask[:, :4, :, :] 

                reco_loss = l1_distance(reco, imgs)
                regression_loss = l1_distance(z_out_tilde, z_out)

                sf_seg = F.softmax(seg_pred, dim=1)
                dice_loss_lv = losses.dice_loss(sf_seg[:,0,:,:], true_masks[:,0,:,:])
                dice_loss_myo = losses.dice_loss(sf_seg[:,1,:,:], true_masks[:,1,:,:])
                dice_loss_rv = losses.dice_loss(sf_seg[:,2,:,:], true_masks[:,2,:,:])
                dice_loss_bg = losses.dice_loss(sf_seg[:, 3, :, :], true_masks[:, 3, :, :])
                loss_dice = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg 

                ce_target = ce_mask[:, 3, :, :]*0 + ce_mask[:, 0, :, :]*1 + ce_mask[:, 1, :, :]*2 + ce_mask[:, 2, :, :]*3 
                seg_pred_swap = torch.cat((seg_pred[:,3,:,:].unsqueeze(1), seg_pred[:,:3,:,:]), dim=1) 
                loss_focal = focal(seg_pred_swap, ce_target) 

                d_cls = criterion(cls_out, labels) + criterion(domain_cls_1, labels)
                d_losses = d_cls * config['DLS']

                batch_loss = prior_loss + reco_loss + (config['RECON'] * regression_loss) + 5*loss_dice + 5*loss_focal + d_losses + config["SSL"] * ssl_loss

                total_meta_loss += batch_loss

                writer.add_scalar('Meta_train_Loss/loss_dice', loss_dice.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_lv', dice_loss_lv.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_myo', dice_loss_myo.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_rv', dice_loss_rv.item(), global_step)
                writer.add_scalar('Meta_train_Loss/loss_focal', loss_focal.item(), global_step)
                writer.add_scalar('Meta_train_Loss/loss_d_cls', d_cls.item(), global_step)
                writer.add_scalar('Meta_train_Loss/loss_ssl', ssl_loss.item(), global_step)
                writer.add_scalar('Meta_train_Loss/batch_loss', batch_loss.item(), global_step)


                # meta-test: 1. load meta-test data 2. calculate meta-test loss
                ###############################Meta test#######################################################
                imgs = meta_test_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = meta_test_masks.clone().to(device=device, dtype=torch.long)
                true_masks = meta_test_masks.to(device=device, dtype=mask_type)
                labels = meta_test_labels.to(device=device, dtype=torch.float32)
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,_,_,domain_cls_1 = model(imgs, labels, 'training', meta_loss=batch_loss)

                ssl_loss = contra_loss(fea_sets)

                reco_loss = l1_distance(reco, imgs)
                regression_loss = l1_distance(z_out_tilde, z_out)

                seg_pred = mask[:, :4, :, :]
                sf_seg = F.softmax(seg_pred, dim=1)
                dice_loss_lv = losses.dice_loss(sf_seg[:,0,:,:], true_masks[:,0,:,:])
                dice_loss_myo = losses.dice_loss(sf_seg[:,1,:,:], true_masks[:,1,:,:])
                dice_loss_rv = losses.dice_loss(sf_seg[:,2,:,:], true_masks[:,2,:,:])
                dice_loss_bg = losses.dice_loss(sf_seg[:, 3, :, :], true_masks[:, 3, :, :])
                loss_dice = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

                ce_target = ce_mask[:, 3, :, :]*0 + ce_mask[:, 0, :, :]*1 + ce_mask[:, 1, :, :]*2 + ce_mask[:, 2, :, :]*3

                seg_pred_swap = torch.cat((seg_pred[:,3,:,:].unsqueeze(1), seg_pred[:,:3,:,:]), dim=1)
                loss_focal = focal(seg_pred_swap, ce_target) 

                d_cls = criterion(domain_cls_1, labels) + criterion(cls_out, labels)
                d_losses = d_cls * config['DLS']

                batch_loss = 5*loss_dice + 5*loss_focal + reco_loss + d_losses + config['SSL'] * ssl_loss + config['RECON']*regression_loss

                total_meta_loss += batch_loss

                optimizer.zero_grad()
                optimizer2.zero_grad()
                total_meta_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                optimizer2.step()

                pbar.set_postfix(**{'loss (batch)': total_meta_loss.item()})
                pbar.update(imgs.shape[0])

                if (epoch + 1) > (k1) and (epoch + 1) % k2 == 0: # 大于30，且是3的倍数
                    if global_step % ((long_len//batch_sz) // 2) == 0:
                        a_feature = F.softmax(mask, dim=1)
                        a_feature = latent_norm(a_feature)
                        # writer.add_images('Meta_train_images/train', imgs, global_step)
                        # writer.add_images('Meta_train_images/a_out0', a_feature[:, 0, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out1', a_feature[:, 1, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out2', a_feature[:, 2, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out3', a_feature[:, 3, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out4', a_feature[:, 4, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out5', a_feature[:, 5, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out6', a_feature[:, 6, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/a_out7', a_feature[:, 7, :, :].unsqueeze(1), global_step)
                        # writer.add_images('Meta_train_images/train_reco', reco, global_step)
                        # writer.add_images('Meta_train_images/train_true', true_masks[:,0:3,:,:], global_step)
                        # writer.add_images('Meta_train_images/train_pred', sf_seg[:,0:3,:,:] > 0.5, global_step)
                        # writer.add_images('Meta_test_images/train_un_img', un_imgs, global_step)

                global_step += 1

            if optimizer.param_groups[0]['lr']<=2e-8:
                print('Converge')
            if (epoch + 1) > k1 and (epoch + 1) % k2 == 0: 
                # validation 
                val_score, val_lv, val_myo, val_rv = eval_dgnet(model, val_loader, device, mode='val')
                scheduler.step(val_score) 
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                logging.info('Validation Dice Coeff: {}'.format(val_score))
                logging.info('Validation LV Dice Coeff: {}'.format(val_lv))
                logging.info('Validation MYO Dice Coeff: {}'.format(val_myo))
                logging.info('Validation RV Dice Coeff: {}'.format(val_rv))

                writer.add_scalar('Dice/val', val_score, epoch)
                writer.add_scalar('Dice/val_lv', val_lv, epoch)
                writer.add_scalar('Dice/val_myo', val_myo, epoch)
                writer.add_scalar('Dice/val_rv', val_rv, epoch)

                initial_itr = 0
                for imgs, true_masks,_ in test_loader:
                    if initial_itr == 5:
                        model.eval()
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        with torch.no_grad():
                            mask, reco, z_out, z_out_tilde, cls_out, fea_sets,_,_,_ = model(imgs, None, 'test')
                        seg_pred = mask[:, :4, :, :]
                        mask_type = torch.float32
                        true_masks = true_masks.to(device=device, dtype=mask_type)
                        sf_seg_pred = F.softmax(seg_pred, dim=1)
                        writer.add_images('Test_images/test', imgs, epoch)
                        writer.add_images('Test_images/test_reco', reco, epoch)
                        writer.add_images('Test_images/test_true', true_masks[:, 0:3, :, :], epoch)
                        writer.add_images('Test_images/test_pred', sf_seg_pred[:, 0:3, :, :] > 0.5, epoch)
                        model.train()
                        break
                    else:
                        pass
                    initial_itr += 1

                # test
                test_score, test_lv, test_myo, test_rv = eval_dgnet(model, test_loader, device, mode='test') 

                if best_dice < test_score:
                    best_dice = test_score
                    best_lv = test_lv
                    best_myo = test_myo
                    best_rv = test_rv
                    print("Epoch checkpoint")
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(model,
                               dir_checkpoint + '/CP_{}.pth'.format(epoch))
                    logging.info('Checkpoint saved !')
                else:
                    pass
                logging.info('Best Dice Coeff: {}'.format(best_dice))
                logging.info('Best LV Dice Coeff: {}'.format(best_lv))
                logging.info('Best MYO Dice Coeff: {}'.format(best_myo))
                logging.info('Best RV Dice Coeff: {}'.format(best_rv))
                writer.add_scalar('Dice/test', test_score, epoch)
                writer.add_scalar('Dice/test_lv', test_lv, epoch)
                writer.add_scalar('Dice/test_myo', test_myo, epoch)
                writer.add_scalar('Dice/test_rv', test_rv, epoch)
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

