#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import cv2
import torch
import matplotlib.pyplot as plt
from hsvnet.data.dataset import TrackingLine


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from hsvnet.data.data_augment import random_spot_augmentation_torch

    trackingline_dataset = TrackingLine(ann_name="val.json", data_dir=r"D:\hsvnet\dataset\raicom")
    trackingline_loader = DataLoader(trackingline_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0")
    best_iou = 0
    best_thr = 0

    miou_list = []
    for i, (x, y, w, h) in enumerate(trackingline_loader):
        x = x[:, :, :int(h[0]), :int(w[0])].to(device)
        y = y[:, :int(h[0]), :int(w[0])].to(device)

        gray = (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]) / 3

        predict = (gray < 73).type(torch.uint8)
        target = y.type(torch.uint8)

        intersection = predict & target
        union = predict | target

        miou = torch.sum(intersection) / torch.sum(union)
        miou_list.append(miou.item())

    miou = np.mean(miou_list)
    print(miou)

