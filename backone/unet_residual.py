# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019-12-28 15:52
# @Author   : Fabrice LI
# @File     : unet_residual.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, dilation=1, padding=1, downsample=None, batchnorm=None):
        super(BasicBlock, self).__init__()
        # 是否有自定义batch norm
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                               bias=False)
        self.bn1 = batchnorm(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                               bias=False)
        self.bn2 = batchnorm(out_planes)
        if downsample is None:
            if in_planes != out_planes:
                downsample = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                    batchnorm(out_planes)
                )
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


class Down(nn.Module):
    def __init__(self, in_planes, out_planes, batchnorm):
        super(Down, self).__init__()
        downsample = None
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d
        if in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                batchnorm(out_planes)
            )
        self.basic_net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            BasicBlock(in_planes, out_planes, downsample=downsample, batchnorm=batchnorm)
        )

    def forward(self, x):
        x = self.basic_net(x)
        return x


class Up(nn.Module):
    def __init__(self, in_planes, out_planes, batchnorm, bilinear=True):
        super(Up, self).__init__()
        downsample = None
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d
        if in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                batchnorm(out_planes)
            )
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2)
        self.conv = BasicBlock(in_planes, out_planes, downsample=downsample, batchnorm=batchnorm)

    def forward(self, x, x_crop):
        x = self.up(x)
        x = self.center_crop(x, x_crop)

        x = torch.cat([x, x_crop], dim=1)

        x = self.conv(x)
        return x

    def center_crop(self, x1, x2):
        # torch image: C X H X W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        return F.pad(x1, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])


class UNetResidual(nn.Module):
    def __init__(self, in_planes, out_planes, n_class, padding, bilinear=True):
        super(UNetResidual, self).__init__()

        self.in_planes = in_planes
        self.out_planed = out_planes
        self.padding = padding
        self.bilinear = bilinear
        self.n_class = n_class

        self.conv_in = BasicBlock(in_planes, 64)
        self.down1 = Down(64, 128, batchnorm=None)
        self.down2 = Down(128, 256, batchnorm=None)
        self.down3 = Down(256, 512, batchnorm=None)
        self.down4 = Down(512, 1024, batchnorm=None)
        self.up1 = Up(1024, 512, batchnorm=None, bilinear=bilinear)
        self.up2 = Up(512, 256, batchnorm=None, bilinear=bilinear)
        self.up3 = Up(256, 128, batchnorm=None, bilinear=bilinear)
        self.up4 = Up(128, 64, batchnorm=None, bilinear=bilinear)
        self.conv_out = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x)

        out = self.conv_out(up4)
        return out


if __name__ == '__main__':
    model = UNetResidual(3, 2, 8, 0, bilinear=False)
    input = torch.rand(1, 3, 572, 572)
    output = model(input)
    print(output.size())
