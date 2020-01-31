# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-18 14:24
# @Author   : Fabrice LI
# @File     : train.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

from config import Config
from models.DeeplabV3Plus import DeeplabV3Plus
from models.backbones.UNet import UNet
from utils.dataset import Apolloscapes
from utils.evaluator import Evaluator
from utils.loss import CELoss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.tensorboard_summary import TensorboardSummary


class Train(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # 初始化tensorboard summary
        self.summary = TensorboardSummary(directory=args.save_path)
        self.writer = self.summary.create_summary()
        # 初始化dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_dataset = Apolloscapes('train_dataset.csv', '/home/aistudio/data/data1919/Image_Data', '/home/aistudio/data/data1919/Gray_Label',
                                     args.crop_size, type='train')

        self.dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

        self.val_dataset = Apolloscapes('val_dataset.csv', '/home/aistudio/data/data1919/Image_Data', '/home/aistudio/data/data1919/Gray_Label',
                                          args.crop_size, type='val')

        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)

        # 初始化model
        self.model = DeeplabV3Plus(backbone=args.backbone,
                              output_stride=args.out_stride,
                              batch_norm=args.batch_norm,
                              num_classes=args.num_classes,
                              pretrain=True)
        # self.model = UNet(in_planes=3, n_class=args.num_classes, padding=1, bilinear=False, pretrain=False)
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         # momentum=args.momentum,
                                         # nesterov=args.nesterov,
                                         weight_decay=args.weight_decay,
                                         lr=args.lr)

        # 定义损失函数
        self.loss = CELoss(num_class=args.num_classes, cuda=args.cuda)

        # 定义验证器
        self.evaluator = Evaluator(args.num_classes)

        # 定义学习率
        self.scheduler = LR_Scheduler('poly', args.lr, args.epochs, len(self.dataloader))

        # 使用cuda
        if args.cuda:
            self.model = self.model.cuda(device=args.gpus[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
        self.best_pred = 0.0

    def train(self, epoch, trainF):
        loss = 0.0
        self.model.train()
        data = tqdm(self.dataloader)
        length = len(self.dataloader)
        for i, sample_list in enumerate(data):
            sample = sample_list[0]
            image_path = sample_list[1]
            label_path = sample_list[2]
            image, label = sample['image'], sample['label']
            if self.args.cuda:
                image = image.cuda(device=self.args.gpus[0])
                label = label.cuda(device=self.args.gpus[0])
            self.scheduler(self.optimizer, i, epoch, 0.0)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss_function = self.loss(output, label)
            loss_function.backward()
            self.optimizer.step()
            loss += loss_function.item()
            # Add batch sample into evaluator
            pred = output.data.cpu().numpy()
            target = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)
            data.set_description('Train loss: %.3f' % (loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss_function.item(), i + length * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (length // 10) == 0:
                global_step = i + length * epoch
                self.summary.visualize_image(self.writer, image, label, output, global_step)
            self.writer.add_scalar('train/total_loss_epoch', loss, epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % loss)
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            print('train miou: %.3f' % mIoU)
            new_pred = mIoU
            if epoch != 0:
                if float(new_pred) - float(self.best_pred) >= 0.5 and i < self.args.batch_size:
                    trainF.write("{}, {}, new_pred: {}, best_pred: {} \n".format(image_path[i], label_path[i],
                                                                                 new_pred, self.best_pred))
                    trainF.flush()


    def val(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample_list in enumerate(tbar):
            sample = sample_list[0]
            # image_path = sample_list[1]
            # label_path = sample_list[2]
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(device=self.args.gpus[0]), target.cuda(device=self.args.gpus[0])
            with torch.no_grad():
                output = self.model(image)
            loss = self.loss(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="BaiDu Lane Segmentation Training")
    parser.add_argument('--backbone', default=Config.BACKBONE, type=str,
                        choices=['resnet', 'hrnet', 'mobilenet', 'unet', 'xception'],
                        help='backbone default resnet101')
    parser.add_argument('--out-stride', default=Config.OUTPUT_STRIDE, type=int,
                        help='backbone output image stride default 16')
    parser.add_argument('--workers', default=Config.NUM_WORKERS, type=int, metavar='N',
                        help='dataloader threads')
    parser.add_argument('--crop-size', default=Config.CROP_SIZE, type=list, help='crop image size')
    parser.add_argument('--sync-bn', default=Config.SYNC_BN, type=bool,
                        help='whether to use sync batch normalization default auto')
    parser.add_argument('--loss', default=Config.LOSS, type=str, choices=['bce', 'dice', 'bce+dice', 'focal'],
                        help='loss function type default bce+dice')
    parser.add_argument('--save-path', default=Config.SAVE_PATH, type=str, help='where to save the model and logs')
    parser.add_argument('--num-classes', default=Config.NUM_CLASSES, type=int, help='number classes to output')
    parser.add_argument('--batch-norm', default=Config.BATCHNORM, type=int, help='batch norm type default 0-frn')

    # hyper parameters
    parser.add_argument('--batch-size', default=Config.BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--epochs', default=Config.EPOCHS, type=int, help='numbers of epochs to training default 8')

    # opt parameters
    parser.add_argument('--lr', default=Config.BASE_LR, type=float, metavar='LR', help='learning rate default 0.001')
    parser.add_argument('--nesterov', default=Config.NESTEROV, action='store_true',
                        help='whether to use nesterov default False')  # store_ture只要运行时该变量有传参就将该变量设为True。
    parser.add_argument('--momentum', default=Config.MOMENTUM, type=float, help='momentum default 0.9')
    parser.add_argument('--weight-decay', default=Config.WEIGHT_DECAY, type=float, help='weight decay default 5e-4')

    # cuda log
    parser.add_argument('--no-cuda', default=Config.NO_CUDA, action='store_true', help='disable cuda')
    parser.add_argument('--gpus', default=Config.DEVICES, type=list, help='use which gpu to train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.sync_bn is None:
        if args.cuda and len(args.gpus) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    trainer = Train(args)
    trainF = open(os.path.join(args.save_path, "train.csv"), 'w')
    for epoch in range(args.epochs):
        trainer.train(epoch, trainF)
        trainer.val(epoch)
    trainer.writer.close()
    trainF.close()


if __name__ == '__main__':
    main()
