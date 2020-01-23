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

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from .dataset import decode_color_labels


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_color_labels(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3,
                               normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_color_labels(torch.squeeze(target[:3], 1).detach().cpu().numpy()), 3,
                               normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
