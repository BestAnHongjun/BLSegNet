#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import os

import torch.nn as nn
from blsegnet.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.print_interval = 10
        self.eval_interval = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = r"datasets/raicom"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.spot_prob = 1.0
        self.max_light_val = 100

    def get_model(self):

        def init_blsegnet(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        from blsegnet.model import BLSegNet, BLSegHead, CSPDarknet
        in_channels = [256, 512, 1024]
        # NANO model use depthwise = True without PAFPN structure, which is main difference.
        backbone = CSPDarknet(self.depth, self.width, depthwise=True, act=self.act)
        head = BLSegHead(self.width, in_channels, depthwise=True, act=self.act)
        self.model = BLSegNet(backbone, head, depthwise=True, pafpn=False)

        self.model.apply(init_blsegnet)
        self.model.head.initialize_biases(1e-2)
        return self.model
