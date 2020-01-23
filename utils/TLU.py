# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-21 10:56
# @Author   : yukkyo
# @File     : TLU.py
# @User     : yukkyo
# @Software : PyCharm
# @Description: https://github.com/yukkyo/PyTorch-FilterResponseNormalizationLayer
#Reference:**********************************************
import torch.nn as nn
import torch


class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)
