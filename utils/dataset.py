# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-23 10:49
# @Author   : Fabrice LI
# @File     : dataset.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import random
import torch

from PIL import Image, ImageFilter
from torch.utils import data
import pandas as pd
import numpy as np
from torchvision.transforms import transforms


class Apolloscapes(data.Dataset):
    def __init__(self, csv_file, imgs_dir, masks_dir, crop_size, type='train'):
        super().__init__()
        self.csv = csv_file
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.train_list = pd.read_csv(csv_file)
        self.crop_size = crop_size
        self.type = type

    def __getitem__(self, item):
        image_path = self.train_list['image'][item]
        label_path = self.train_list['label'][item]

        image = Image.open(image_path).convert('RGB')
        label = np.array(Image.open(label_path), dtype=np.uint8)

        _tmp = encode_labels(label)
        label = Image.fromarray(_tmp)
        sample = {'image': image, 'label':label}
        if self.type == 'train':
            sample = transform_train(sample, self.crop_size)
        elif self.type == 'test':
            sample = transform_test(sample, self.crop_size)
        elif self.type == 'val':
            sample = transform_val(sample, self.crop_size)
        elif not self.type:
            return sample, image_path, label_path
        else:
            raise NotImplementedError
        return sample

    def __len__(self):
        return len(self.train_list)


def transform_train(sample, crop_size):
    composed_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        ScaleCrop(crop_size=crop_size),
        RandomGaussianBlur(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


def transform_test(sample, crop_size):
    composed_transforms = transforms.Compose([
        ScaleCrop(crop_size=crop_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


def transform_val(sample, crop_size):
    composed_transforms = transforms.Compose([
        ScaleCrop(crop_size=crop_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask}


class ScaleCrop(object):
    # offset 690 需要heatmap统计
    def __init__(self, crop_size=[768, 256], offset=690):
        self.crop_size = crop_size
        self.offset = offset

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img_w, img_h = img.size
        mask_w, mask_h = mask.size
        img = img.crop((0, self.offset, img_w, img_h))
        mask = mask.crop((0, self.offset, mask_w, mask_h))

        img = img.resize((self.crop_size[0], self.crop_size[1]), Image.BILINEAR)
        mask = mask.resize((self.crop_size[0], self.crop_size[1]), Image.NEAREST)

        return {'image': img, 'label': mask}


def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]))
    # 0
    encode_mask[color_mask == 0] = 0
    encode_mask[color_mask == 249] = 0
    encode_mask[color_mask == 255] = 0
    # 1
    encode_mask[color_mask == 200] = 1
    encode_mask[color_mask == 204] = 1
    encode_mask[color_mask == 213] = 0
    encode_mask[color_mask == 209] = 1
    encode_mask[color_mask == 206] = 0
    encode_mask[color_mask == 207] = 0
    # 2
    encode_mask[color_mask == 201] = 2
    encode_mask[color_mask == 203] = 2
    encode_mask[color_mask == 211] = 0
    encode_mask[color_mask == 208] = 0
    # 3
    encode_mask[color_mask == 216] = 0
    encode_mask[color_mask == 217] = 3
    encode_mask[color_mask == 215] = 0
    # 4 In the test, it will be ignored
    encode_mask[color_mask == 218] = 0
    encode_mask[color_mask == 219] = 0
    # 4
    encode_mask[color_mask == 210] = 4
    encode_mask[color_mask == 232] = 0
    # 5
    encode_mask[color_mask == 214] = 5
    # 6
    encode_mask[color_mask == 202] = 0
    encode_mask[color_mask == 220] = 6
    encode_mask[color_mask == 221] = 6
    encode_mask[color_mask == 222] = 6
    encode_mask[color_mask == 231] = 0
    encode_mask[color_mask == 224] = 6
    encode_mask[color_mask == 225] = 6
    encode_mask[color_mask == 226] = 6
    encode_mask[color_mask == 230] = 0
    encode_mask[color_mask == 228] = 0
    encode_mask[color_mask == 229] = 0
    encode_mask[color_mask == 233] = 0
    # 7
    encode_mask[color_mask == 205] = 7
    encode_mask[color_mask == 212] = 0
    encode_mask[color_mask == 227] = 7
    encode_mask[color_mask == 223] = 0
    encode_mask[color_mask == 250] = 7

    return encode_mask


def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    deocde_mask[labels == 0] = 0
    # 1
    deocde_mask[labels == 1] = 204
    # 2
    deocde_mask[labels == 2] = 203
    # 3
    deocde_mask[labels == 3] = 217
    # 4
    deocde_mask[labels == 4] = 210
    # 5
    deocde_mask[labels == 5] = 214
    # 6
    deocde_mask[labels == 6] = 224
    # 7
    deocde_mask[labels == 7] = 227

    return deocde_mask


def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    data_dir = '../data/train_dataset.csv'

    apollo_val = Apolloscapes(data_dir, '../data/Image_Data', '../data/Gray_Label', [1024, 384], type='train')

    dataloader = DataLoader(apollo_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_color_labels(tmp)
            segmap = np.transpose(segmap, axes=[1, 2, 0])
            segmap = segmap.astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
