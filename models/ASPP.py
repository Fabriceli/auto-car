# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-21 14:57
# @Author   : Fabrice LI
# @File     : ASPP.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import torch.nn as nn
import torch
import torch.nn.functional as F

from config import OUTPUT_CHANNEL_BACKBONE, OUTPUT_STRIDE_BACKBONE, BatchNorm
from utils.FRN import FilterResponseNorm2d
from utils.TLU import TLU
from utils.sync_batchnorm import SynchronizedBatchNorm2d


class ASPP(nn.Module):
    def __init__(self, backbone, batch_norm):
        super(ASPP, self).__init__()
        self.inplanes = OUTPUT_CHANNEL_BACKBONE[backbone]
        if not self.inplanes:
            raise NotImplementedError
        # 对于backbone输出图像尺寸是输入的1/8的情况，aspp的膨胀卷积的rate的选择
        if OUTPUT_STRIDE_BACKBONE[backbone] == 8:
            dilations = [1, 12, 24, 36]
        elif OUTPUT_STRIDE_BACKBONE[backbone] == 16:
            dilations = [1, 6, 12, 18]
        else:
            raise NotImplementedError
        if batch_norm == BatchNorm.FRN:
            BN = FilterResponseNorm2d
            LU = TLU
        elif batch_norm == BatchNorm.SYNC:
            BN = SynchronizedBatchNorm2d
            LU = nn.ReLU
        else:
            BN = nn.BatchNorm2d
            LU = nn.ReLU
        # 1x1 Conv
        self.conv1 = nn.Conv2d(self.inplanes, 256, kernel_size=1, padding=0, dilation=dilations[0], bias=False)
        self.bn1 = BN(256)
        self.tlu1 = LU(256)
        # 3x3 conv rate 6
        self.conv2 = nn.Conv2d(self.inplanes, 256, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.bn2 = BN(256)
        self.tlu2 = LU(256)
        # 3x3 conv rate 12
        self.conv3 = nn.Conv2d(self.inplanes, 256, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False)
        self.bn3 = BN(256)
        self.tlu3 = LU(256)
        # 3x3 conv rate 18
        self.conv4 = nn.Conv2d(self.inplanes, 256, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False)
        self.bn4 = BN(256)
        self.tlu4 = LU(256)
        # image pooling
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(self.inplanes, 256, kernel_size=1, bias=False),
                                     BN(256),
                                     LU(256))

        # last 1x1 conv
        self.conv5 = nn.Conv2d(5*256, 256, kernel_size=1, bias=False)
        self.bn5 = BN(256)
        self.tlu5 = LU(256)

        self.__weight_init()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.tlu1(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.tlu2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.tlu3(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.tlu4(x4)

        x5 = self.avgpool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # pytorch tensor N C H W,dim=1表示channel维度进行sum
        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x7 = self.conv5(x6)
        x7 = self.bn5(x7)
        x7 = self.tlu5(x7)

        return x7

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


def build_aspp(backbone, batch_norm):
    return ASPP(backbone, batch_norm)


if __name__ == '__main__':
    pass
