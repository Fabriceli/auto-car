# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-23 17:43
# @Author   : Fabrice LI
# @File     : tensorboard_summary.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import os
import torch

import numpy as np
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from .dataset import decode_labels, decode_color_labels


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
    def get_color_mask(self, pred):
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = torch.argmax(pred, dim=0)
        pred = torch.squeeze(pred)
        pred = pred.detach().cpu().numpy()
        pred = decode_labels(pred)
        pred = torch.from_numpy(pred)
    
        return pred
    def visualize_image(self, writer, image, ground_truth, output_model, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(self.get_color_mask(output_model), 3,
                               normalize=False, range=(0, 255))
        writer.add_image('Predicted', grid_image, global_step)
        print(ground_truth.shape)
        grid_image = make_grid(ground_truth, 3, normalize=False, range=(0, 255))
        writer.add_image('Ground truth', grid_image, global_step)
        
        
