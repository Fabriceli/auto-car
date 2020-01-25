# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-18 20:08
# @Author   : Fabrice LI
# @File     : loss.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import torch.nn.functional as F
import torch.nn as nn
import torch


class CELoss(nn.Module):
    def __init__(self, num_class, weight=None, ignore_label=255, cuda=True):
        super(CELoss, self).__init__()
        self.weight = weight
        self.ignore_label = ignore_label
        self.num_class = num_class
        self.cuda = cuda

    def forward(self, predict, label):
        # if predict.dim() > 2:
        #     predict = predict.view(predict.size(0), predict.size(1), -1)  # N,C,H,W => N,C,H*W
        #     predict = predict.transpose(1, 2)  # N,C,H*W => N,H*W,C
        #     predict = predict.contiguous().view(-1, self.num_class)  # N,H*W,C => N*H*W,C
        # label = label.view(-1)
        # 交叉熵loss默认使用均值模式，reduction='mean'
        loss = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_label)
        if self.cuda:
            loss = loss.cuda()
        return loss(predict, label.long())


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, label):
        assert predict.shape == label.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(label.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], label[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == label.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(label.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/label.shape[1]


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(inputs,dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at
        # mask = mask.view(-1)
        loss = -1 * (1 - pt) ** self.gamma * logpt #* mask
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    predict = torch.rand(1, 8, 384, 384).cpu()
    label = torch.rand(1, 384, 384).cpu()
    celoss = CELoss(num_class=8)
    diceloss = DiceLoss()
    focalloss = FocalLoss()
    print(celoss(predict, label).item())
    # print(diceloss(predict, label).item())
    print(focalloss(predict, label.long()).item())

