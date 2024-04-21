#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from blsegnet.data.data_augment import preproc, preproc_mask


class TrackingLine(Dataset):
    def __init__(self,
                 ann_name,
                 data_dir="datasets",
                 input_size=(416, 416),
                 ):
        self.ann_path = os.path.join(data_dir, ann_name)
        self.img_path = os.path.join(data_dir, "Images")
        self.input_size = input_size
        
        self.coco = COCO(self.ann_path)
        imgIds = self.coco.getImgIds(catIds=[0])
        self.imgInfo = self.coco.loadImgs(imgIds)
    
    def __getitem__(self, index):
        imgInfo = self.imgInfo[index]
        imPath = os.path.join(self.img_path, imgInfo["file_name"])
        img = cv2.imread(imPath)
        img, truth_width, truth_height = preproc(img, self.input_size)
        annIds = self.coco.getAnnIds(imgIds=imgInfo['id'])
        anns = self.coco.loadAnns(annIds)
        mask = self.coco.annToMask(anns[0])
        mask, _, _ = preproc_mask(mask, self.input_size)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.int)
        return img_tensor, mask_tensor, truth_width, truth_height
    
    def __len__(self):
        return len(self.imgInfo)


# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from torch.utils.data import DataLoader
#     from blsegnet.data.data_augment import random_spot_augmentation_torch

#     trackingline_dataset = TrackingLine(ann_name="train.json", data_dir=r"D:\hsvnet\dataset\cnsoftbei")
#     trackingline_loader = DataLoader(trackingline_dataset, batch_size=1, shuffle=True)

#     device = torch.device("cuda:0")
#     for i, (x, y, w, h) in enumerate(trackingline_loader):
#         x = x.to(device)
#         y = y.to(device)

#         aug_img = random_spot_augmentation_torch(x, device, h, w, 100).cpu().numpy()
#         h = int(h[0])
#         w = int(w[0])
#         aug_img = np.transpose(aug_img, (0, 2, 3, 1)).astype(np.uint8)[:, :h, :w, :]

#         test_img = x.cpu().numpy()
#         test_img = np.transpose(test_img, (0, 2, 3, 1)).astype(np.uint8)[:, :h, :w, :]

#         test_mask = y.cpu().numpy()[:, :h, :w]

#         batch_size = x.shape[0]
#         for batch in range(batch_size):
#             cv2.imwrite(os.path.join(r"D:\hsvnet\aug_output", "{}.src.jpg".format(i)), test_img[batch])
#             cv2.imwrite(os.path.join(r"D:\hsvnet\aug_output", "{}.aug.jpg".format(i)), aug_img[batch])
#             cv2.imwrite(os.path.join(r"D:\hsvnet\aug_output", "{}.light.jpg".format(i)), aug_img[batch] - test_img[batch])
#             cv2.imwrite(os.path.join(r"D:\hsvnet\aug_output", "{}.seg.jpg".format(i)), test_mask[batch] * 255)

#             aug_img_b = cv2.cvtColor(aug_img[batch], cv2.COLOR_BGR2RGB)
#             test_img_b = cv2.cvtColor(test_img[batch], cv2.COLOR_BGR2RGB)

#             plt.subplot(batch_size, 4, batch * 4 + 1)
#             plt.title("source-image")
#             plt.imshow(test_img_b)
#             plt.subplot(batch_size, 4, batch * 4 + 3)
#             plt.title("augment-image")
#             plt.imshow(aug_img_b)
#             plt.subplot(batch_size, 4, batch * 4 + 2)
#             plt.title("random-spot-light")
#             plt.imshow(aug_img_b - test_img_b)
#             plt.subplot(batch_size, 4, batch * 4 + 4)
#             plt.title("segment-mask")
#             plt.imshow(test_mask[batch] * 255)

#         plt.show()
#         print(x.shape, y.shape, w, h)
