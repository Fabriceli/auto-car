# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019-12-27 15:05
# @Author   : Fabrice LI
# @File     : resnet.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from utils.sync_batchnorm import SynchronizedBatchNorm2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, dilation=1, downsample=None, batchnorm=None):
        super(BasicBlock, self).__init__()
        # 是否有自定义batch norm
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=False)
        self.bn1 = batchnorm(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=False)
        self.bn2 = batchnorm(out_planes)
        self.downsample = downsample

    def forward(self, x):
        x_identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # 是否需要调整x的维度，调整后才能和开始的x相加
        if self.downsample is not None:
            x_identity = self.downsample(x_identity)
        out = x + x_identity

        out = self.relu(out)

        return out


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, dilation=1, downsample=None, batchnorm=None):
        super(BottleBlock, self).__init__()
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = batchnorm(out_planes)
        self.relu = nn.ReLU(inplace=True)
        # 当dilation的时候padding=dilation
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = batchnorm(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = batchnorm(out_planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        x_identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            x_identity = self.downsample(x_identity)
        out = x + x_identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, out_stride, n_class=1000, batchnorm=None, pretrained=True):
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        # out_stride表示输出图像大小是原图像的多少分之一，例如out_stride=16，表示输出图像是原图的16分之一
        # dilatedFCN的思路，添加膨胀卷积，输出图像大小不变，感受野增大
        if out_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif out_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        if batchnorm is None:
            batchnorm = nn.BatchNorm2d
        self.batchnorm = batchnorm
        self.in_planes = 64
        # padding=3 是为了same mode
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.batchnorm(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # padding=1 同上 same mode
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # print("layer0: " + str(self.in_planes))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       batchnorm=batchnorm)
        # print("layer1: " + str(self.in_planes))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       batchnorm=batchnorm)
        # print("layer2: " + str(self.in_planes))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       batchnorm=batchnorm)
        # print("layer3: " + str(self.in_planes))
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
        #                                batchnorm=batchnorm)
        # print("layer4: " + str(self.in_planes))

        # 对于嵌入deeplab的backbone下面需要修改，todo 可以提升部分 why？
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, n_class)
        self.layer4 = self._make_MG_unit(block, 512, blocks, stride=strides[3], dilation=dilations[3],
                                         batchnorm=batchnorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, batchnorm=None):
        # block: 类型，是basicblock还是bottleblock
        # blocks：block数量
        downsample = None
        if batchnorm is None:
            batchnorm = self.batchnorm
        # 调整维度
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                batchnorm(planes * block.expansion)
            )
        # 开始构造layer
        layers = []
        # 第一次block需要调整stride，第二次后的不需要
        layers.append(block(self.in_planes, planes, stride, dilation, downsample, batchnorm))
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation, batchnorm=batchnorm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # 论文第九页Table 2里面说到是在Conv2后作为输出是最有效的， 作为decoder的输入
        low_level = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x, low_level

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, batchnorm=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, batchnorm=batchnorm))
        self.in_planes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.in_planes, planes, stride=1,
                                dilation=blocks[i]*dilation, batchnorm=batchnorm))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(out_stride, batchnorm):
    model = ResNet(BottleBlock, [3, 4, 23, 3], out_stride, batchnorm=batchnorm)
    return model


if __name__ == '__main__':
    import torch

    model = ResNet101(batchnorm=nn.BatchNorm2d, out_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
