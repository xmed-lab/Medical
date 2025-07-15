#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 15:08
# @Author  : xxxxxx
# @Site    : 
# @File    : main_others.py
# @Software: PyCharm

# #Desc: main for rvf dataset
#### 务必仔细检查meta_prompt是否修正为加法, 还有prompt num的大小,还有meta-decoder 9->16; 还有reconstruct变化力求显存


import os
import gc
import torch
import logging
import models
import utils
import random
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
from load_others import get_seg_dg_dataloader
from data_others import train_dg_collate_fn, test_collate_fn, test_dg_collate_fn
from torch.utils.tensorboard import SummaryWriter
from eval_others import eval_dgnet
from metrics import FocalLoss, ContraLoss
from utils import KL_divergence
from torchmetrics.classification import BinaryF1Score

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



def set_seed(seed=42):
    # 设置PyTorch随机种子
    torch.manual_seed(seed)

    # 设置随机种子以确保数据加载的可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置随机种子以确保NumPy操作的可重复性
    np.random.seed(seed)

    # 设置随机种子以确保Python内置随机模块的可重复性
    random.seed(seed)







def prepare_data(cfg, batch_sz, test_vendor):
    """这里在预处理时候没有操作，所以需要shuffle"""
    train_domains, test_domain = list(set([0,1,2,3]) - set([test_vendor])), [test_vendor]

    print("Train domain: {}, Test domain: {}".format(train_domains, test_domain))
    # 创造 dataloader
    cfg['TRAIN'] = [train_domains[0]]
    cfg['TEST'] = test_domain
    print("CFG",cfg)
    domain_1_labeled_dataset, _, _ = get_seg_dg_dataloader(cfg, batch_sz, workers=1, reid=0)
    cfg['TRAIN'] = [train_domains[1]]
    domain_2_labeled_dataset,_, _ = get_seg_dg_dataloader(cfg, batch_sz, workers=1, reid=1)
    cfg['TRAIN'] = [train_domains[2]]
    domain_3_labeled_dataset,_, _ = get_seg_dg_dataloader(cfg, batch_sz, workers=1, reid=2)

    _,_, test_loader = get_seg_dg_dataloader(cfg, batch_sz, workers=1, reid=3) # 有没有不打紧reid

    d_len = []
    d_len.append(len(domain_1_labeled_dataset))
    d_len.append(len(domain_2_labeled_dataset))
    d_len.append(len(domain_3_labeled_dataset))


    # Label data balance 必须要balance到最大的，否则无法循环；
    long_len = max(d_len) # max dataset
    print('Max len of source domains ', long_len)
    print('Before balance: Domain 1,2,3: ', d_len)
    new_d_1 = domain_1_labeled_dataset
    for i in range(long_len // d_len[0]):
        if long_len == d_len[0]:
            break
        new_d_1 = ConcatDataset([new_d_1, domain_1_labeled_dataset])
    domain_1_labeled_dataset = new_d_1
    domain_1_labeled_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=2, pin_memory=False, collate_fn=train_dg_collate_fn)

    new_d_2 = domain_2_labeled_dataset
    for i in range(long_len // d_len[1]):
        if long_len == d_len[1]:
            break
        new_d_2 = ConcatDataset([new_d_2, domain_2_labeled_dataset])
    domain_2_labeled_dataset = new_d_2
    domain_2_labeled_loader = DataLoader(dataset=domain_2_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=2, pin_memory=False, collate_fn=train_dg_collate_fn)

    new_d_3 = domain_3_labeled_dataset
    for i in range(long_len // d_len[2]):
        if long_len == d_len[2]:
            break
        new_d_3 = ConcatDataset([new_d_3, domain_3_labeled_dataset])
    domain_3_labeled_dataset = new_d_3
    domain_3_labeled_loader = DataLoader(dataset=domain_3_labeled_dataset, batch_size=batch_sz // 2, shuffle=True,
                                         drop_last=False, num_workers=2, pin_memory=False, collate_fn=train_dg_collate_fn) # 这个最好和imagefolder只留一个，不然有可能会造成死锁；

    print("After balance Domain=1,2,3: ", len(domain_1_labeled_dataset), len(domain_2_labeled_dataset), len(domain_3_labeled_dataset))
    print("Len of balanced labeled dataloader: Domain 1: {}, Domain 2: {}, Domain 3: {}.".\
          format(len(domain_1_labeled_loader), len(domain_2_labeled_loader), len(domain_3_labeled_loader)))

    return long_len, domain_1_labeled_loader, domain_2_labeled_loader, domain_3_labeled_loader, test_loader


def train(config):
    # params
    best_dice, best_1, best_2, best_3 = 0, 0, 0, 0
    epochs = config['EPOCH']
    batch_sz = config['BATCH']
    lr = config['LR']
    test_vendor = config['TASK']
    device = config['device']
    dir_checkpoint = os.path.join(config['LOG'], 'ckps', config['MODEL'] + '-' + config['DATA'] + '-' + config['TASK'] +'-' + str(config['RATIO'])+'lora')
    if not os.path.exists(dir_checkpoint):
        print("create the ckpt dirs")
        os.makedirs(dir_checkpoint)
    wc = os.path.join(config['LOG'], 'logdirs', config['MODEL'] + '-' + config['DATA'] + '-' + config['TASK'] +'-' +str(config['RATIO'])+'lora') # tensorboard
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
        'num_mask_channels': 2}
    global_params = {
        "prompt_size": 90, # no promot_seg; 90
        "domain_num" : 3,}
    encoder_params = {
        "pretrained": True,
        "pretrained_vit_name": 'vit_base_patch16_224_in21k',
        "pretrained_folder" : r'/home/czhaobo/Medical/src/models',
        "img_size": config['CROP_SIZE'],
        "num_frames" : 8,
        "patch_size" : 16,
        "in_chans" : 3,
        "embed_dim" : 768,
        "depth" : 12,# full transunet会不会有问题
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
    # print('Decoder Parameters: ', utils.count_parameters(model.decoder), utils.count_parameters(model.decoder.seg_head), utils.count_parameters(model.decoder.recon_head))

    model.to(device)


    # data set
    cfg = {}
    cfg['DATASET'] = config['DATASET']
    cfg['ROOT'] = config['ROOT']

    long_len, domain_1_labeled_loader,domain_2_labeled_loader,domain_3_labeled_loader,test_loader = prepare_data(cfg, batch_sz, int(test_vendor))


    # metric & criterion
    criterion = nn.CrossEntropyLoss().to(device)
    l1_distance = nn.L1Loss().to(device)
    contra_loss = ContraLoss(config['TEMP'])
    bce_loss = torch.nn.BCELoss()

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
    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-6) # 不要限制的苔丝
    optimizer2 = optim.AdamW(params_to_optimize2, lr=lr, weight_decay=1e-6)

    if config['CKPS']:
        print("load params: ")
        # model = torch.load(checkpoint[])
        checkpoint = torch.load(config['CKPS'])
        model = checkpoint['model_state_dict']
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])

    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=opt_patience) # 指定在降低学习率之前要等待多少个指标评估周期。如果在 patience 个周期内指标没有改善，学习率将被减小。
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'max', factor=0.5, patience=opt_patience) # 指定在降低学习率之前要等待多少个指标评估周期。如果在 patience 个周期内指标没有改善，学习率将被减小。
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

            # num iter: iteration
            for num_itr in range(long_len//batch_sz): # 不用除2的原因是因为meta_test会复制两次
                # Randomly choosing meta train and meta test domains every iters; 其实这也就相当于meta-learning的support set和eval set
                domain_list = np.random.permutation(3)
                meta_train_domain_list = domain_list[:2]
                meta_test_domain_list = domain_list[2]

                meta_train_imgs, meta_train_masks, meta_train_labels = [], [], []
                meta_test_imgs, meta_test_masks, meta_test_labels = [], [], []
                sample = next(domain_labeled_iter_list[meta_train_domain_list[0]])
                imgs, true_masks, labels = sample['image'], sample['label'], sample['dc'].squeeze()
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)
                sample = next(domain_labeled_iter_list[meta_train_domain_list[1]])
                imgs, true_masks, labels = sample['image'], sample['label'], sample['dc'].squeeze()
                meta_train_imgs.append(imgs)
                meta_train_masks.append(true_masks)
                meta_train_labels.append(labels)

                sample = next(domain_labeled_iter_list[meta_test_domain_list]) # why twice
                imgs, true_masks, labels = sample['image'], sample['label'], sample['dc'].squeeze()
                meta_test_imgs.append(imgs) # why twice;因为要匹配两个域,而且因为labeled的dataset的batch_size//2了
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)
                sample = next(domain_labeled_iter_list[meta_test_domain_list])
                imgs, true_masks, labels = sample['image'], sample['label'], sample['dc'].squeeze()
                meta_test_imgs.append(imgs)
                meta_test_masks.append(true_masks)
                meta_test_labels.append(labels)


                meta_train_imgs = torch.cat((meta_train_imgs[0], meta_train_imgs[1]), dim=0) # 两个meta-source域
                meta_train_masks = torch.cat((meta_train_masks[0], meta_train_masks[1]), dim=0)
                meta_train_labels = torch.cat((meta_train_labels[0], meta_train_labels[1]), dim=0)
                meta_test_imgs = torch.cat((meta_test_imgs[0], meta_test_imgs[1]), dim=0) # meta_test这里其实也对应两个
                meta_test_masks = torch.cat((meta_test_masks[0], meta_test_masks[1]), dim=0)
                meta_test_labels = torch.cat((meta_test_labels[0], meta_test_labels[1]), dim=0)


                # train preprocess
                # supervise loss
                total_meta_loss = 0.0
                # meta-train: 1. load meta-train data 2. calculate meta-train loss
                ###############################Meta train#######################################################
                imgs = meta_train_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = meta_train_masks.clone().to(device=device, dtype=torch.long)
                true_masks = meta_train_masks.to(device=device, dtype=mask_type)
                labels = meta_train_labels.to(device=device, dtype=torch.float32)

                mask, reco, z_out, z_out_tilde, cls_out, fea_sets, mu, logvar, domain_cls_1= model(imgs, labels, 'training')

                kl_loss1 = KL_divergence(logvar[:, :8], mu[:, :8])
                kl_loss2 = KL_divergence(logvar[:, 8:], mu[:, 8:])
                prior_loss = (kl_loss1 + kl_loss2) * config['prior_reg']


                ssl_loss = contra_loss(fea_sets)

                seg_pred = mask[:, :1, :, :] # 选前2个

                reco_loss =  0# .01 * l1_distance(reco, imgs)
                regression_loss = 0#l1_distance(z_out_tilde, z_out)
                sf_seg = F.sigmoid(seg_pred)
                dc_loss = losses.dice_loss(sf_seg[:, 0, :, :], true_masks[:, 0, :, :])

                dice_loss_1 = bce_loss(sf_seg[:, 0, :, :], true_masks[:, 0, :, :])
                loss_dice = dice_loss_1 + dc_loss



                loss_focal = 0
                d_cls = criterion(cls_out, labels) + criterion(domain_cls_1, labels)
                d_losses = d_cls * config['DLS']

                batch_loss = prior_loss + reco_loss + (config['RECON'] * regression_loss) + 5*loss_dice + 5*loss_focal + d_losses + config["SSL"] * ssl_loss

                total_meta_loss += batch_loss


                writer.add_scalar('Meta_train_Loss/loss_dice', loss_dice.item(), global_step)
                writer.add_scalar('Meta_train_Loss/dice_loss_1', dice_loss_1.item(), global_step)
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
                mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar, domain_cls_1 = model(imgs, labels, 'training', meta_loss=batch_loss
                                                                          ) # batch_loss

                ssl_loss = contra_loss(fea_sets)

                reco_loss = 0 # .01 * l1_distance(reco, imgs)
                regression_loss =0 # l1_distance(z_out_tilde, z_out)

                seg_pred = mask[:, :1, :, :]
                sf_seg = F.sigmoid(seg_pred)
                dc_loss = losses.dice_loss(sf_seg[:, 0, :, :], true_masks[:, 0, :, :])
                dice_loss_1 = bce_loss(sf_seg[:, 0, :, :], true_masks[:, 0, :, :])
                loss_dice = dice_loss_1  + dc_loss# + dsc  # + dice_loss_2 + dice_loss_3

                loss_focal = 0 #  focal(seg_pred_swap, ce_target) # 前面是background，不需要似乎。

                d_cls = criterion(cls_out, labels) + criterion(domain_cls_1, labels)
                d_losses = d_cls * config['DLS']

                batch_loss = 5*loss_dice + 5*loss_focal + reco_loss + d_losses + config['SSL'] * ssl_loss + config['RECON']*regression_loss
                # print("A ",5*loss_dice,5*loss_focal, reco_loss, d_losses ,config['SSL'] * ssl_loss , config['RECON']*regression_loss)

                total_meta_loss += batch_loss

                optimizer.zero_grad()
                optimizer2.zero_grad()
                total_meta_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                optimizer2.step()

                # scheduler.step()


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
                        # writer.add_images('Meta_train_images/train_true', true_masks[:,0:1,:,:], global_step)
                        # writer.add_images('Meta_train_images/train_pred', sf_seg[:,0:1,:,:] > 0.5, global_step)
                        # writer.add_images('Meta_test_images/train_un_img', un_imgs, global_step)

                global_step += 1

            if optimizer.param_groups[0]['lr']<=2e-8:
                print('Converge')
            if (epoch + 1) % k2 == 0: # 偶数才会记录，艹
                initial_itr = 0
                for i, sample in enumerate(test_loader): # 生成这样的图片
                    if initial_itr == 1:
                        model.eval()
                        imgs, true_masks = sample['image'], sample['label']

                        imgs = imgs.to(device=device, dtype=torch.float32)
                        with torch.no_grad():
                            mask, reco, z_out, z_out_tilde, cls_out, fea_sets,mu,logvar,_= model(imgs, None, 'test')
                        seg_pred = mask[:, :1, :, :] # 2 for optic
                        mask_type = torch.float32
                        true_masks = true_masks.to(device=device, dtype=mask_type)
                        sf_seg_pred = torch.sigmoid(seg_pred)

                        # writer.add_images('Test_images/test', imgs, epoch)
                        # writer.add_images('Test_images/test_reco', reco, epoch)
                        # writer.add_images('Test_images/test_true', true_masks[:, 0:1, :, :], epoch)
                        # writer.add_images('Test_images/test_pred', sf_seg_pred[:, 0:1, :, :] > 0.5, epoch)
                        model.train()
                        break
                    else:
                        pass
                    initial_itr += 1

                # 测试
                test_score = eval_dgnet(model, test_loader, device, mode='test') # 这里才是测试得分

                # if epoch == 61:
                #     torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': model,
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'optimizer2_state_dict': optimizer2.state_dict(),
                #     }, dir_checkpoint + '/CP_{}.pth'.format(epoch))

                if best_dice < test_score: # 只汇报目前最好情况
                    best_dice = test_score

                    print("Epoch checkpoint")
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(model,
                               dir_checkpoint + '/CP_{}.pth'.format(epoch))

                    # torch.save({
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'optimizer2_state_dict': optimizer2.state_dict(),
                    # }, dir_checkpoint + '/CP_{}.pth'.format(epoch))

                    logging.info('Checkpoint saved !')
                else:
                    pass
                print('best', best_dice)

                logging.info('Best Dice Coeff: {}'.format(best_dice))
                # writer.add_scalar('Dice/test', test_score, epoch)

            gc.collect()
            torch.cuda.empty_cache()

        writer.close()




if __name__ == '__main__':
    # set_seed(2013)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    config = config
    device = torch.device('cuda:'+str(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu')
    config['device'] = device
    logging.info(f'Using device {device}')

    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)

    train(config)


