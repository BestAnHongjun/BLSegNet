#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import os

import torch.nn as nn
from hsvnet.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.print_interval = 10
        self.eval_interval = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = r"D:\hsvnet\dataset\cnsoftbei"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

    def get_model(self):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        from hsvnet.model import HSVNet, HSVHead, PAFPN
        in_channels = [256, 512, 1024]
        # MINI model use depthwise = True, which is main difference.
        backbone = PAFPN(self.depth, self.width, depthwise=True, act=self.act)
        head = HSVHead(self.width, in_channels, depthwise=True, act=self.act)
        self.model = HSVNet(backbone, head, depthwise=True, pafpn=True)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
