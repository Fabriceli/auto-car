# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2020-01-18 14:26
# @Author   : Fabrice LI
# @File     : config.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
from enum import Enum, unique


@unique
class Backbone(Enum):
    RESNET = 0
    MOBILENET = 1
    XCEPTION = 2

@unique
class BatchNorm(Enum):
    FRN = 0
    SYNC = 1


class Config(object):
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8
    BACKBONE = Backbone.RESNET
    CROP_SIZE = [768, 256]

    # train config
    EPOCHS = 8
    WEIGHT_DECAY = 5.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 0.001
    NUM_WORKERS = 4
    DEVICES = [2]
    BATCH_SIZE = 4
    SYNC_BN = None
    LOSS = 'bce+dice'
    BATCHNORM = BatchNorm.FRN

    # opt
    NESTEROV = False
    MOMENTUM = 0.9

    # cuda
    NO_CUDA = False


OUTPUT_STRIDE_BACKBONE = {
    Backbone.RESNET: 16,
    Backbone.XCEPTION: 16,
    Backbone.MOBILENET: 8
}
OUTPUT_CHANNEL_BACKBONE = {
    Backbone.RESNET: 2048,
    Backbone.XCEPTION: 2048,
    Backbone.MOBILENET: 320
}

BATCH_NORM = {
    'frn': 0,
    'syn': 1
}

LOW_LEVEL_BACKBONE = {
    Backbone.RESNET: 256,
    Backbone.XCEPTION: 128,
    Backbone.MOBILENET: 24
}
