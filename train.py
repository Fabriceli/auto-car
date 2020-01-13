# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019-12-26 21:14
# @Author   : Fabrice LI
# @File     : train.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import argparse
import os
import shutil
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from Config import Config
from datasets.basic_dataset import ApolloScape
from model.deeplab import DeepLab


def generate_matrix(gt_image, pre_image, num_class):
    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


# Compute Mean Iou
def mean_iou(pred, label, num_classes=8):
    confusion_matrix = generate_matrix(label, pred, num_classes)
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def compute_iou(pred, gt, result):
    """
    pred : [N, H, W]
    gt: [N, H, W]
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_gt = gt == i
        single_pred = pred == i
        temp_tp = np.sum(single_gt * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result


def create_loss(predict, label, num_classes):
    # BCE with DICE
    bce_loss = nn.CrossEntropyLoss(predict, label)
    loss = bce_loss
    miou = mean_iou(predict, label, num_classes)
    return loss, miou


def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['label']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=config.DEVICES[0]), mask.cuda(device=config.DEVICES[0])
        optimizer.zero_grad()
        out = net(image)
        mask_loss = nn.CrossEntropyLoss()(out, mask.long())
        total_mask_loss += mask_loss.item()
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("train epoch:{}".format(epoch))
        dataprocess.set_postfix_str("train mask_loss:{:.4f}".format(mask_loss.item()))
    trainF.write("train epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['label']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=config.DEVICES[0]), mask.cuda(device=config.DEVICES[0])
        out = net(image)
        mask_loss = nn.CrossEntropyLoss()(out, mask.long())
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("test epoch:{}".format(epoch))
        dataprocess.set_postfix_str("test mask_loss:{:.4f}".format(mask_loss))
    testF.write(" Test Epoch:{} \n".format(epoch))
    for i in range(8):
        result_string = "miou: {}: {:.4f} \n".format(i, result["TP"][i] / result["TA"][i])
        print(result_string)
        testF.write(result_string)
    testF.write("test epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()


def main():
    lane_config = Config()
    if os.path.exists(lane_config.SAVE_PATH):
        shutil.rmtree(lane_config.SAVE_PATH)
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)

    trainF = open(os.path.join(lane_config.SAVE_PATH, "train.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "test.csv"), 'w')

    # Define Dataloader
    train_data_dir = 'data_list/train_dataset.csv'
    val_data_dir = 'data_list/val_dataset.csv'

    # Get data list and split it into train and validation set.
    image_data_dir = 'data/Image_Data'
    label_data_dir = 'data/Gray_Label'

    kwargs = {'num_workers': lane_config.NUM_WORKERS, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_data = ApolloScape(train_data_dir, image_data_dir, label_data_dir, transform='train')

    val_data = ApolloScape(val_data_dir, image_data_dir, label_data_dir, transform='val')

    train_loader = DataLoader(train_data, batch_size=lane_config.BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)

    val_loader = DataLoader(val_data, batch_size=lane_config.BATCH_SIZE, shuffle=False, drop_last=False, **kwargs)

    # Define network
    model = DeepLab(num_classes=lane_config.NUM_CLASSES,
                    backbone=lane_config.BACKBONE,
                    output_stride=lane_config.OUTPUT_STRIDE,
                    sync_bn=True,
                    freeze_bn=False)

    if torch.cuda.is_available():
        model = model.cuda(device=lane_config.DEVICES[0])
        model = torch.nn.DataParallel(model, device_ids=lane_config.DEVICES)

    optimizer = torch.optim.Adam(model.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    for epoch in range(lane_config.EPOCHS):
        train_epoch(model, epoch, train_loader, optimizer, trainF, lane_config)
        test(model, epoch, val_loader, testF, lane_config)
        if epoch % 2 == 0:
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
    trainF.close()
    testF.close()
    torch.save({'state_dict': model.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))


if __name__ == "__main__":
    main()
