# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-06 20:22
# @Author   : Fabrice LI
# @File     : Config.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
class Config(object):
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8
    BACKBONE = 'resnet'

    # train config
    EPOCHS = 8
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 0.0006
    NUM_WORKERS = 4
    DEVICES = [0]
    BATCH_SIZE = 8
