#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import os

from blsegnet.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (416, 416)
        self.print_interval = 10
        self.eval_interval = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = r"datasets/raicom"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.spot_prob = 1.0
        self.max_light_val = 100
