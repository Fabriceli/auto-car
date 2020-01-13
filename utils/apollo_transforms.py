# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019-12-27 00:45
# @Author   : Fabrice LI
# @File     : apollo_transforms.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import random

import torch
import numpy as np
from PIL import Image, ImageFilter


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}


class ScaleCrop(object):
    # offset 690 需要heatmap统计
    def __init__(self, crop_size=[1024, 384], offset=690):
        self.crop_size = crop_size
        self.offset = offset

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img_w, img_h = img.size
        mask_w, mask_h = mask.size
        img = img.crop((0, self.offset, img_w, img_h))
        mask = mask.crop((0, self.offset, mask_w, mask_h))

        img = img.resize((self.crop_size[0], self.crop_size[1]), Image.BILINEAR)
        mask = mask.resize((self.crop_size[0], self.crop_size[1]), Image.NEAREST)

        return {'image': img, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask}


# 有什么用？
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img, 'label': mask}
