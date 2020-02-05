# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-21 20:33
# @Author   : Fabrice LI
# @File     : UNet.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torch

from config import BatchNorm
from utils.FRN import FilterResponseNorm2d
from utils.TLU import TLU


class BasicNet(nn.Module):
    def __init__(self, in_planes, out_planes, padding, batch_norm=BatchNorm.FRN):
        super(BasicNet, self).__init__()
        if batch_norm == BatchNorm.FRN:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=padding),
                FilterResponseNorm2d(out_planes),
                TLU(out_planes),
                nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=padding),
                FilterResponseNorm2d(out_planes),
                TLU(out_planes)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=padding),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=padding),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_planes, out_planes, padding):
        super(Down, self).__init__()
        self.basic_net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            BasicNet(in_planes, out_planes, padding)
        )

    def forward(self, x):
        x = self.basic_net(x)
        return x


class Up(nn.Module):
    def __init__(self, in_planes, out_planes, padding, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2)
        self.conv = BasicNet(in_planes, out_planes, padding)

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


class UNet(nn.Module):
    def __init__(self, in_planes, n_class, padding, bilinear=True, pretrain=True):
        super(UNet, self).__init__()
        self.in_planes = in_planes
        self.padding = padding
        self.bilinear = bilinear
        self.n_class = n_class

        self.conv_in = BasicNet(in_planes, 64, padding)
        self.down1 = Down(64, 128, padding)
        self.down2 = Down(128, 256, padding)
        self.down3 = Down(256, 512, padding)
        self.down4 = Down(512, 1024, padding)
        self.up1 = Up(1024, 512, padding, bilinear=bilinear)
        self.up2 = Up(512, 256, padding, bilinear=bilinear)
        self.up3 = Up(256, 128, padding, bilinear=bilinear)
        self.up4 = Up(128, 64, padding, bilinear=bilinear)
        self.conv_out = nn.Conv2d(64, n_class, kernel_size=1)

        self.weight_init()
        if pretrain:
            self.pretrain()

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

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pretrain(self):
        # pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        # pretrain_dict = torch.load('/home/aistudio/work/auto-car/models/mobilenet_v2.pth')
        pretrain_dict = torch.load('/home/aistudio/work/auto-car/models/models/mobilenet_0.3749_768x256.pth.tar')['state_dict']
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    import torch

    model = UNet(3, 8, 1, bilinear=False, pretrain=False)
    input = torch.rand(1, 3, 572, 572)
    output = model(input)
    print(output.size())
