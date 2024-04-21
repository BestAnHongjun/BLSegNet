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

    trackingline_dataset = TrackingLine(ann_name="val.json", data_dir=r"D:\hsvnet\dataset\cnsoftbei")
    trackingline_loader = DataLoader(trackingline_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda:0")
    best_iou = 0
    best_thr = 0

    miou_list = []
    for i, (x, y, w, h) in enumerate(trackingline_loader):
        x = x[0, :, :int(h[0]), :int(w[0])].numpy().transpose(1, 2, 0).astype(np.uint8)
        y = y[0, :int(h[0]), :int(w[0])].numpy().astype(np.uint8) * 255

        cv2.imwrite("dataset_output/cnsoftbei/{}.src.jpg".format(i), x)
        cv2.imwrite("dataset_output/cnsoftbei/{}.mask.jpg".format(i), y)

        print(i)
