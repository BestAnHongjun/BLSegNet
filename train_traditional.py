#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import torch
import matplotlib.pyplot as plt
from hsvnet.data.dataset import TrackingLine


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from hsvnet.data.data_augment import random_spot_augmentation_torch

    fig = plt.figure()
    # fig.subplots_adjust(left=0.04, right=0.96, top=0.925, bottom=0.055, hspace=0.2, wspace=0.15)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    trackingline_dataset = TrackingLine(ann_name="train.json", data_dir=r"D:\hsvnet\dataset\cnsoftbei")
    trackingline_loader = DataLoader(trackingline_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0")
    best_iou = 0
    best_thr = 0

    for thr in range(256):
        miou_list = []
        for i, (x, y, w, h) in enumerate(trackingline_loader):
            x = x[:, :, :int(h[0]), :int(w[0])].to(device)
            y = y[:, :int(h[0]), :int(w[0])].to(device)

            gray = (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]) / 3

            predict = (gray < thr).type(torch.uint8)
            target = y.type(torch.uint8)

            intersection = predict & target
            union = predict | target

            miou = torch.sum(intersection) / torch.sum(union)
            miou_list.append(miou.item())
        miou = np.mean(miou_list)
        if miou > best_iou:
            best_thr = thr
            best_iou = miou
        print("thr:{}, miou:{:.4f}%, best_miou:{:.4f}%, best_thr:{}".format(thr, miou * 100, best_iou * 100, best_thr))

