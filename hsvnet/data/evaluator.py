#!/usr/bin/env python3
# Copyright (c) Coder.AN. All rights reserved.

import time
import torch
from loguru import logger
from tqdm import tqdm
import numpy as np


class Evaluator:

    def __init__(self, dataloader, seg_threshold: float):
        self.dataloader = dataloader
        self.seg_threshold = seg_threshold

    def evaluate(self, model, device, half=False):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        miou_list = []

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, targets, w, h) in enumerate(tqdm(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.to(device)
                targets = targets.to(device)
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                if is_time_record:
                    infer_end = time.time()
                    inference_time += infer_end - start

                assert outputs.shape == targets.shape

                batch_size = targets.shape[0]
                miou = 0

                for batch in range(batch_size):
                    outputs_b = outputs[batch, :h[batch], :w[batch]]
                    targets_b = targets[batch, :h[batch], :w[batch]]

                    outputs_b = outputs_b > self.seg_threshold
                    outputs_b = outputs_b.type(torch.int8)
                    targets_b = targets_b.type(torch.int8)
                    intersection = outputs_b & targets_b
                    union = outputs_b | targets_b
                    miou += (torch.sum(intersection) / torch.sum(union)).item()

                miou /= batch_size
                miou_list.append(miou)

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])

        eval_results = self.evaluate_prediction(miou_list, statistics)

        return eval_results

    def evaluate_prediction(self, miou_list, statistics):
        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        n_samples = statistics[1].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["inference"],
                    [a_infer_time],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(miou_list) > 0:
            miou = np.mean(miou_list)
            info += "mIoU: {:.4f}\n".format(miou)
            return miou, info
        else:
            return 0, info
