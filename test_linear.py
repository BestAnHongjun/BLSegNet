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
    trackingline_loader = DataLoader(trackingline_dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda:0")
    for i, (x, y, w, h) in enumerate(trackingline_loader):
        x = x.to(device)
        y = y.to(device)

        test_img = x[0].cpu().numpy()
        test_img = np.transpose(test_img, (1, 2, 0)).astype(np.uint8)[:int(h[0]), :int(w[0]), :]

        test_mask = y[0].cpu().numpy()[:int(h[0]), :int(w[0])]

        black = test_img[test_mask == 0]
        white = test_img[test_mask == 1]

        ax.scatter(black[:, 0], black[:, 1], black[:, 2], label='background', c='red', s=20)
        ax.scatter(white[:, 0], white[:, 1], white[:, 2], label='road', c='blue', s=20)

        plt.show()

        exit(0)
