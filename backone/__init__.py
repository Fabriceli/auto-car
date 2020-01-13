# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2019-12-26 19:48
# @Author   : Fabrice LI
# @File     : __init__.py.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
from backone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
