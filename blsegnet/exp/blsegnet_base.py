#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import os

import torch
import torch.nn as nn

from .base_exp import BaseExp

__all__ = ["Exp", "check_exp_value"]


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "lrelu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = None
        # name of annotation file for training
        self.train_ann = "train.json"
        # name of annotation file for evaluation
        self.val_ann = "val.json"

        # ---------------- device config ---------------- #
        # set id of cuda device
        self.cuda_id = 0

        # --------------- transform config ----------------- #
        # prob of applying random spot aug
        self.spot_prob = 1.0
        self.max_light_val = 100

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 100
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "hsvnetwarmcos"
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # segmentation threshold
        self.seg_threshold = 0.5

    def get_model(self):
        from blsegnet.model import BLSegNet, PAFPN, BLSegHead

        def init_blsegnet(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = PAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = BLSegHead(self.width, in_channel=in_channels, act=self.act)
            self.model = BLSegNet(backbone, head)

        self.model.apply(init_blsegnet)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self):
        """
        Get dataset according to cache and cache_type parameters.
        """
        from blsegnet.data.dataset import TrackingLine

        return TrackingLine(
            ann_name=self.train_ann,
            data_dir=self.data_dir,
            input_size=self.input_size,
        )

    def get_data_loader(self, batch_size):
        """
        Get dataloader according to cache_img parameter.
        """
        from torch.utils.data.dataloader import DataLoader

        if self.dataset is None:
            self.dataset = self.get_dataset()

        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.data_num_workers,
            pin_memory=True
        )

        return train_loader

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from blsegnet.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_dataset(self):
        from blsegnet.data.dataset import TrackingLine

        return TrackingLine(
            ann_name=self.val_ann,
            data_dir=self.data_dir,
            input_size=self.input_size,
        )

    def get_eval_loader(self, batch_size):
        from torch.utils.data.dataloader import DataLoader

        val_dataset = self.get_eval_dataset()

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.data_num_workers,
            pin_memory=True
        )

        return val_loader

    def get_evaluator(self, batch_size):
        from blsegnet.data import Evaluator

        return Evaluator(
            dataloader=self.get_eval_loader(batch_size),
            seg_threshold=self.seg_threshold,
        )

    def get_trainer(self, args):
        from blsegnet.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, device, half=False):
        return evaluator.evaluate(model, device, half)


def check_exp_value(exp: Exp):
    h, w = exp.input_size
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"