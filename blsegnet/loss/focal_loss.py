#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

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

            target_b = target_b.type(torch.int)
            # assert isinstance(target, torch.IntTensor) or isinstance(target, torch.cuda.IntTensor)

            pt = torch.zeros_like(predict_b)
            pt[target_b == 1] = predict_b[target_b == 1]
            pt[target_b == 0] = 1 - predict_b[target_b == 0]

            loss = -torch.pow(1 - pt, self.gamma) * torch.log(pt + self.eps)
            mean_loss += loss.sum() / loss.numel()

        mean_loss /= batch_size
        return mean_loss
