#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import math
import torch
import torch.nn as nn
from .network_blocks import BaseConv, DWConv


class HSVHead(nn.Module):
    def __init__(self, width=1.0, in_channel=[256, 512, 1024], depthwise=False, act="lrelu"):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv
        self.hsv_conv = Conv(
            int(in_channel[2] * width), 3, 1, 1, act=act, bias=True
        )
        self.upsample = nn.Upsample(scale_factor=32, mode="nearest")
        self.sigmoid = nn.Sigmoid()
        self.depthwise = depthwise

    def forward(self, fpn_feature, x):
        hsv_feature = self.hsv_conv(fpn_feature)
        hsv_feature_expand = self.upsample(hsv_feature)
        assert hsv_feature_expand.shape == x.shape
        linear_output = self.sigmoid(hsv_feature_expand[:, 0, :, :] * x[:, 0, :, :]
                         + hsv_feature_expand[:, 1, :, :] * x[:, 1, :, :]
                         + hsv_feature_expand[:, 2, :, :] * x[:, 2, :, :])
        return linear_output

    def initialize_biases(self, prior_prob):
        if self.depthwise:
            b = self.hsv_conv.dconv.conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            self.hsv_conv.dconv.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b = self.hsv_conv.pconv.conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            self.hsv_conv.pconv.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        else:
            b = self.hsv_conv.conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            self.hsv_conv.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
