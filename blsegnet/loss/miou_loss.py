#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import torch
import torch.nn as nn


class MIoULoss(nn.Module):
    def __init__(self, seg_threshold=0.5):
        super().__init__()
        self.seg_threshold = seg_threshold

    def forward(self, predict, target, max_height=None, max_width=None):
        assert predict.shape == target.shape
        if max_height is None:
            max_height = torch.zeros_like(target)
            max_height[:] = target.shape[1]
        if max_width is None:
            max_width = torch.zeros_like(target)
            max_width[:] = target.shape[2]

        batch_size = target.shape[0]
        mean_loss = 0

        for batch in range(batch_size):
            predict_b = predict[batch, :max_height[batch], :max_width[batch]]
            target_b = target[batch, :max_height[batch], :max_width[batch]]

            predict_b = predict_b > self.seg_threshold
            predict_b = predict_b.type(torch.int8)
            target_b = target_b.type(torch.int8)
            intersection = predict_b & target_b
            union = predict_b | target_b

            mean_loss += torch.sum(intersection) / torch.sum(union)

        mean_loss /= batch_size
        return 1 - mean_loss
