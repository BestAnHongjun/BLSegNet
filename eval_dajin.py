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
    trackingline_loader = DataLoader(trackingline_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda:0")
    best_iou = 0
    best_thr = 0

    miou_list = []
    for i, (x, y, w, h) in enumerate(trackingline_loader):
        x = x[:, :, :int(h[0]), :int(w[0])].to(device)
        y = y[:, :int(h[0]), :int(w[0])].to(device)

        gray = (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]) / 3
        gray_cv = gray.cpu().numpy()[0].astype(np.uint8)

        _, ret = cv2.threshold(gray_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        predict = np.zeros_like(ret)
        predict[ret == 0] = 1
        predict[ret == 255] = 0

        target = y.cpu().numpy()[0].astype(np.uint8)

        intersection = predict & target
        union = predict | target

        miou = np.sum(intersection) / np.sum(union)
        miou_list.append(miou.item())

    miou = np.mean(miou_list)
    print(miou)

