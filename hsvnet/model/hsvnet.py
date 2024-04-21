#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import torch.nn as nn

from hsvnet.model.pafpn import PAFPN
from hsvnet.model.hsv_head import HSVHead
from hsvnet.loss import MIoULoss, FocalLoss


class HSVNet(nn.Module):
    def __init__(self, backbone=None, head=None, depthwise=False, pafpn=True):
        super().__init__()
        if backbone is None:
            backbone = PAFPN(depth=0.33, width=0.25, depthwise=depthwise)
        if head is None:
            head = HSVHead(width=0.25, depthwise=depthwise)

        self.backbone = backbone
        self.head = head
        self.depthwise = depthwise
        self.pafpn = pafpn

        self.miou_loss = MIoULoss()
        self.focal_loss = FocalLoss()

    def forward(self, x, targets=None, truth_width=None, truth_height=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)[2] if self.pafpn else self.backbone(x)["dark5"]

        if self.training:
            assert targets is not None
            assert truth_width is not None
            assert truth_height is not None
            predicts = self.head(fpn_outs, x)
            miou_loss = self.miou_loss(predicts, targets, truth_height, truth_width)
            focal_loss = self.focal_loss(predicts, targets, truth_height, truth_width)
            total_loss = miou_loss + focal_loss

            outputs = {
                "total_loss": total_loss,
                "miou_loss": miou_loss,
                "focal_loss": focal_loss,
            }
        else:
            outputs = self.head(fpn_outs, x)

        return outputs
