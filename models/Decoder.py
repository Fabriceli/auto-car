# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-21 14:21
# @Author   : Fabrice LI
# @File     : Decoder.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import torch.nn as nn
import torch
import torch.nn.functional as F

from config import LOW_LEVEL_BACKBONE, BatchNorm
from utils.FRN import FilterResponseNorm2d
from utils.TLU import TLU
from utils.sync_batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, backbone, batch_norm, num_classes):
        super(Decoder, self).__init__()
        if batch_norm == BatchNorm.FRN:
            BN = FilterResponseNorm2d
            LU = TLU
        elif batch_norm == BatchNorm.SYNC:
            BN = SynchronizedBatchNorm2d
            LU = nn.ReLU
        else:
            BN = nn.BatchNorm2d
            LU = nn.ReLU
        self.low_level_inplane = LOW_LEVEL_BACKBONE[backbone]
        self.conv1 = nn.Conv2d(self.low_level_inplane, 48, kernel_size=1, bias=False)
        self.bn1 = BN(48)
        self.tlu1 = LU(48)

        # 48表示low_level经过1x1 conv后的输出维度，256表示encoder经过aspp的输出维度
        # 论文中提到最后卷积层使用了两个卷积效果更好，Ch 4.1
        self.conv2 = nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False)  # n_in = n_out输入输出hw不变
        self.bn2 = BN(256)
        self.tlu2 = LU(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False) # n_in = n_out输入输出hw不变
        self.bn3 = BN(256)
        self.tlu3 = LU(256)

        # 最后
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False)

        # 权重初始化
        self.__weight_init()

    def forward(self, output_encoder, low_level_feature):
        x_low_level = self.conv1(low_level_feature)
        x_low_level = self.bn1(x_low_level)
        x_low_level = self.tlu1(x_low_level)

        # 上采样4倍，以low_level的hw为准，encoder输出的双线性插值扩为low_level的同样大小维度
        x2 = F.interpolate(output_encoder, size=x_low_level.size()[2:], mode='bilinear', align_corners=True)
        # concat
        x3 = torch.cat([x_low_level, x2], dim=1)

        # last conv layer (2 3x3conv)
        x3 = self.conv2(x3)
        x3 = self.bn2(x3)
        x3 = self.tlu2(x3)

        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.tlu3(x3)

        out = self.conv4(x3)
        return out

    def __weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(backbone, batch_norm, num_classes):
    return Decoder(backbone, batch_norm, num_classes)
