# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2020-01-21 20:32
# @Author   : Fabrice LI
# @File     : ResNet.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import torch.nn as nn
import torch
from torch.utils import model_zoo

from config import BatchNorm
from utils.FRN import FilterResponseNorm2d
from utils.TLU import TLU
from utils.sync_batchnorm import SynchronizedBatchNorm2d


class BasicBlock(nn.Module):
    expand_factor = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None, batch_norm=BatchNorm.FRN):
        super(BasicBlock, self).__init__()
        if batch_norm == BatchNorm.FRN:
            BN = FilterResponseNorm2d
            LU = TLU
        elif batch_norm == BatchNorm.SYNC:
            BN = SynchronizedBatchNorm2d
            LU = nn.ReLU
        else:
            BN = nn.BatchNorm2d
            LU = nn.ReLU
        # padding = dilation 使得输入输出维度不变，感受野变大，和原论文里面的stride=2的目的一样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = BN(out_channels)
        self.tlu1 = LU(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = BN(out_channels)
        self.tlu2 = LU(out_channels)

        self.downsample = downsample

    def forward(self, x):
        res = x
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.tlu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        # 判断是否需要调整维度：调整维度
        if self.downsample:
            res = self.downsample(x)
        x3 = x2 + res
        # 最后经过激活函数
        x3 = self.tlu2(x3)
        return x3


class BottleBlock(nn.Module):
    expand_factor = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None, batch_norm=BatchNorm.FRN):
        super(BottleBlock, self).__init__()
        if batch_norm == BatchNorm.FRN:
            BN = FilterResponseNorm2d
            LU = TLU
        elif batch_norm == BatchNorm.SYNC:
            BN = SynchronizedBatchNorm2d
            LU = nn.ReLU
        else:
            BN = nn.BatchNorm2d
            LU = nn.ReLU
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BN(out_channels)
        self.tlu1 = LU(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BN(out_channels)
        self.tlu2 = LU(out_channels)

        # 1x1 conv
        self.conv3 = nn.Conv2d(out_channels, self.expand_factor * out_channels, kernel_size=1, bias=False)
        self.bn3 = BN(self.expand_factor * out_channels)
        self.tlu3 = LU(self.expand_factor * out_channels)
        self.downsample = downsample

    def forward(self, x):
        res = x
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.tlu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.tlu2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        if self.downsample:
            res = self.downsample(res)
        out = x3 + res
        out = self.tlu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, output_stride, batch_norm, nums_block, block_type, pretrain=True):
        super(ResNet, self).__init__()
        if batch_norm == BatchNorm.FRN:
            BN = FilterResponseNorm2d
            LU = TLU
        elif batch_norm == BatchNorm.SYNC:
            BN = SynchronizedBatchNorm2d
            LU = nn.ReLU
        else:
            BN = nn.BatchNorm2d
            LU = nn.ReLU
        self.out_channels = 64
        # 对于输出图像大小是原图的1/16，stride和dilation的配置
        if output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, self.out_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = BN(self.out_channels)
        self.tlu1 = LU(self.out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建卷积层
        self.conv2_x = self._make_layer(block_type, 64, nums_block[0], BN, stride[0], dilation[0])
        self.conv3_x = self._make_layer(block_type, 128, nums_block[1], BN, stride[1], dilation[1])
        self.conv4_x = self._make_layer(block_type, 256, nums_block[2], BN, stride[2], dilation[2])
        self.conv5_x = self._make_layer(block_type, 512, nums_block[3], BN, stride[3], dilation[3])

        # 初始化权重
        self._weight_init()

        # 预训练
        if pretrain:
            self.pre_train()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.tlu1(x1)

        x2 = self.maxpool(x1)

        x3 = self.conv2_x(x2)
        # 经过conv2_x后作为low level的输出会是最有效的，deeplabv3plus论文注
        low_level = x3
        x4 = self.conv3_x(x3)
        x5 = self.conv4_x(x4)
        x6 = self.conv5_x(x5)
        return x6, low_level

    def _make_layer(self, block_type, channel, nums_block, batch_norm, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.out_channels != channel * block_type.expand_factor:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, channel * block_type.expand_factor,
                          kernel_size=1, stride=stride, bias=False),
                batch_norm(channel * block_type.expand_factor)
            )

        layers = []
        layers.append(block_type(self.out_channels, channel, stride, dilation, downsample, batch_norm))
        self.out_channels = channel * block_type.expand_factor
        for i in range(1, nums_block):
            layers.append(block_type(self.out_channels, channel, dilation=dilation, batch_norm=batch_norm))
        return nn.Sequential(*layers)

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pre_train(self):
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        # pretrain_dict = torch.load('/home/aistudio/work/auto-car/models/resnet101.pth')
        pretrain_dict = torch.load('/home/aistudio/work/auto-car/models/laneNet7.pth.tar')['state_dict']
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(output_stride, batch_norm, pretrain):
    return ResNet(output_stride, batch_norm, [3, 4, 23, 3], BottleBlock, pretrain)


if __name__ == '__main__':
    import torch

    model = ResNet101(16, BatchNorm.FRN, pretrain=False)
    input = torch.rand(1, 3, 112, 112)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
