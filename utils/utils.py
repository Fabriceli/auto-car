# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2019-12-27 00:37
# @Author   : Fabrice LI
# @File     : utils.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import numpy as np
from torchvision import transforms
from utils import apollo_transforms as tr


def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    deocde_mask[labels == 0] = 0
    # 1
    deocde_mask[labels == 1] = 204
    # 2
    deocde_mask[labels == 2] = 203
    # 3
    deocde_mask[labels == 3] = 217
    # 4
    deocde_mask[labels == 4] = 219
    # 5
    deocde_mask[labels == 5] = 232
    # 6
    deocde_mask[labels == 6] = 214
    # 7
    deocde_mask[labels == 7] = 221
    # 8
    deocde_mask[labels == 8] = 227

    return deocde_mask


def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]))
    # 0
    encode_mask[color_mask == 0] = 0
    encode_mask[color_mask == 249] = 0
    encode_mask[color_mask == 255] = 0
    # 1
    encode_mask[color_mask == 200] = 1
    encode_mask[color_mask == 204] = 1
    encode_mask[color_mask == 213] = 0
    encode_mask[color_mask == 209] = 1
    encode_mask[color_mask == 206] = 0
    encode_mask[color_mask == 207] = 0
    # 2
    encode_mask[color_mask == 201] = 2
    encode_mask[color_mask == 203] = 2
    encode_mask[color_mask == 211] = 0
    encode_mask[color_mask == 208] = 0
    # 3
    encode_mask[color_mask == 216] = 0
    encode_mask[color_mask == 217] = 3
    encode_mask[color_mask == 215] = 0
    # 4 In the test, it will be ignored
    encode_mask[color_mask == 218] = 0
    encode_mask[color_mask == 219] = 0
    # 4
    encode_mask[color_mask == 210] = 4
    encode_mask[color_mask == 232] = 0
    # 5
    encode_mask[color_mask == 214] = 5
    # 6
    encode_mask[color_mask == 202] = 0
    encode_mask[color_mask == 220] = 6
    encode_mask[color_mask == 221] = 6
    encode_mask[color_mask == 222] = 6
    encode_mask[color_mask == 231] = 0
    encode_mask[color_mask == 224] = 6
    encode_mask[color_mask == 225] = 6
    encode_mask[color_mask == 226] = 6
    encode_mask[color_mask == 230] = 0
    encode_mask[color_mask == 228] = 0
    encode_mask[color_mask == 229] = 0
    encode_mask[color_mask == 233] = 0
    # 7
    encode_mask[color_mask == 205] = 7
    encode_mask[color_mask == 212] = 0
    encode_mask[color_mask == 227] = 7
    encode_mask[color_mask == 223] = 0
    encode_mask[color_mask == 250] = 7

    return encode_mask


def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask


def apollo_transform(sample):
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomGaussianBlur(),
        tr.ScaleCrop(),
        # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)


def val_transform(sample):
    composed_transforms = transforms.Compose([
        tr.ScaleCrop(),
        tr.ToTensor()])

    return composed_transforms(sample)

