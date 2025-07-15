#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/28 8:56
# @Author  : Anonymous
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm

# #Desc: register

from .promptseg import *
# from .doprompt import *
# from .transunet import *
# from .vitwithout import *
# from .prod import *


import sys

def get_model(name, params):
    if name == 'promptransunet':
        return PromptTransUnet(params['width'], params['prompt_dim'],params['head'], params['transformer_dim'], params['z_length'],
                        params['anatomy_out_channels'],params['decoder_type'], params['num_mask_channels'],
                           params["prompt_size"], params["domain_num"],
                         params["pretrained"], params["pretrained_vit_name"],
                         params["pretrained_folder"],
                         params["img_size"], params["num_frames"], params["patch_size"], params["in_chans"], params["embed_dim"],
                        params["depth"], params["num_heads"], params["mlp_ratio"],
                        params["patch_embedding_bias"], params["qkv_bias"], params["qk_scale"], params["drop_rate"], params["attn_drop_rate"],
                        params["drop_path_rate"], params["norm_layer"],
                        params["adapt_method"], params["num_domains"],
                             params["prompt_num"])
    else:
        print("Could not find the requested models ({})".format(name), file=sys.stderr)