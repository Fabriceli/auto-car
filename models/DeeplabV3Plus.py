# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-21 13:48
# @Author   : Fabrice LI
# @File     : DeeplabV3Plus.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import torch.nn as nn
import torch.nn.functional as F

from config import BatchNorm, Backbone
from models.Decoder import build_decoder
from models.Encoder import build_encoder


class DeeplabV3Plus(nn.Module):
    def __init__(self, backbone=Backbone.RESNET, output_stride=16, batch_norm=BatchNorm.FRN, num_classes=8, pretrain=True):
        super(DeeplabV3Plus, self).__init__()
        self.encoder = build_encoder(backbone, output_stride, batch_norm, pretrain)
        self.decoder = build_decoder(backbone, batch_norm, num_classes)

    def forward(self, x_in):
        # low_level_feature, output
        low_level_features, output_encoder = self.encoder(x_in)
        x = self.decoder(output_encoder, low_level_features)
        # decoder输入没有原图大小，所以需要使用双线性插值恢复原图尺寸，其中x是decoder未经过上采样的结果
        x = F.interpolate(x, size=x_in.size()[2:], mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    import torch

    model = DeeplabV3Plus(pretrain=False)
    model.eval()
    input = torch.rand(1, 3, 112, 112)
    output = model(input)
    print(output.size())
