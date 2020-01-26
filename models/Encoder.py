# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-21 14:21
# @Author   : Fabrice LI
# @File     : Encoder.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import torch.nn as nn
from config import Backbone, BatchNorm
from models.ASPP import build_aspp
from models.backbones.MobileNet import MobileNetV2
from models.backbones.ResNet import ResNet101
from models.backbones.Xception import Xception


class Encoder(nn.Module):
    def __init__(self, backbone, output_stride, batch_norm, pretrain):
        super(Encoder, self).__init__()
        if backbone == Backbone.MOBILENET:
            self.backbone = MobileNetV2()
        elif backbone == Backbone.XCEPTION:
            self.backbone = Xception(output_stride, batch_norm, pretrain)
        else:
            self.backbone = ResNet101(output_stride, batch_norm, pretrain)
        self.aspp = build_aspp(backbone, batch_norm)

    def forward(self, input):
        in_aspp, low_level_feature = self.backbone(input)
        output = self.aspp(in_aspp)
        return low_level_feature, output


def build_encoder(backbone, output_stride, batch_norm, pretrain):
    return Encoder(backbone, output_stride, batch_norm, pretrain)


if __name__ == '__main__':
    import torch

    model = Encoder(Backbone.RESNET, 16, BatchNorm.FRN, pretrain=False)
    input = torch.rand(1, 3, 112, 112)
    low_level_feat, output = model(input)
    print(output.size())
    print(low_level_feat.size())
