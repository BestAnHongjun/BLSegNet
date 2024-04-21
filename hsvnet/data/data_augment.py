#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import cv2
import torch
import numpy as np


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    truth_height = int(img.shape[0] * r)
    truth_width = int(img.shape[1] * r)
    resized_img = cv2.resize(
        img,
        (truth_width, truth_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: truth_height, : truth_width] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, truth_width, truth_height


def preproc_mask(mask, input_size):
    if len(mask.shape) == 3:
        padded_mask = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    else:
        padded_mask = np.zeros(input_size, dtype=np.uint8)

    r = min(input_size[0] / mask.shape[0], input_size[1] / mask.shape[1])
    truth_height = int(mask.shape[0] * r)
    truth_width = int(mask.shape[1] * r)
    resized_mask = cv2.resize(
        mask,
        (truth_width, truth_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)
    padded_mask[: truth_height, : truth_width] = resized_mask

    return padded_mask, truth_width, truth_height


def random_spot_augmentation_torch(img: torch.tensor, device, truth_height, truth_width, max_light_val=20):
    if len(img.shape) == 3:
        _, height, width = img.shape
    else:
        _, _, height, width = img.shape

    # 生成随机光斑参数
    batch_size = img.shape[0]
    u_x = torch.randint(low=0, high=height-1, size=(1, )).to(device)
    u_y = torch.randint(low=0, high=width-1, size=(1, )).to(device)
    sig_x = torch.rand(size=(1, )).to(device) * height
    sig_y = torch.rand(size=(1, )).to(device) * width
    p = torch.rand(size=(1, )).to(device) * 2 - 1

    # 光场
    light = torch.zeros_like(img)
    x_mask = torch.zeros_like(light)
    y_mask = torch.zeros_like(light)
    for x in range(height):
        x_mask[:, :, x, :] = x
    for y in range(width):
        y_mask[:, :, :, y] = y
    index = -(torch.pow((x_mask - u_x) / sig_x, 2) -
              2 * p * (x_mask - u_x) * (y_mask - u_y) / sig_x / sig_y +
              torch.pow((y_mask - u_y) / sig_y, 2)) / 2 / (1 - p * p)
    a = 1 / 2 / torch.pi / sig_x / sig_y / torch.sqrt(1 - p * p)
    light += a * torch.exp(index)

    light_val = torch.rand(size=(1,)).to(device) * max_light_val
    if torch.max(light) >= 1e-6:
        light = (light - torch.min(light)) / (torch.max(light) - torch.min(light))
        light *= light_val

    add_mask = torch.zeros_like(img)
    for batch in range(img.shape[0]):
        add_mask[batch, :, : truth_height[batch], : truth_width[batch]] = 1.0
    light *= add_mask

    aug_img = img + light
    aug_img[aug_img > 255] = 255.0

    return aug_img
