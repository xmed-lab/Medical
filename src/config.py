#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 9:37
# @Author  : Anonymous
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc: config

class MedicalConfig():
    """for mms"""
    MODEL = "promptransunet" 
    DATA = "MM"
    TASK = 'C'
    DECODER = 'XXX'
    WEIGHT_INIT = 'xavier'
    CROP_SIZE = 288
    # nohup python main.py >output_B_full.txt 2>&1 & ; jobs -l; ;kill -9;
    # train parameter
    SEED = 2023
    USE_CUDA = True
    GPU = '2'
    EPOCH = 100
    DIM = 64
    LR = 2e-4 
    BATCH = 4
    WD = 0

    # hyper
    TEMP = 0.1
    SSL = 0.01 
    RECON = 0  
    DLS = 1 #1

    # dir parameter
    ROOT = '/home/xxx/Medical/data/'
    LOG = '/home/xxx/Medical/log'

    # ratio
    RATIO = 1

    prior_reg = 1e-4
    meta_step = 1e-2 # 1e-3
    adapter = True



config = vars(MedicalConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
