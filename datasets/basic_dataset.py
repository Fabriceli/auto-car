# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019-12-26 21:14
# @Author   : Fabrice LI
# @File     : basic_dataset.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data

from utils.utils import encode_labels, decode_labels, apollo_transform, val_transform


class ApolloScape(data.Dataset):

    def __init__(self, csv_file, imgs_dir, masks_dir, transform=None):
        super().__init__()
        self.csv = csv_file
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.train_list = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.train_list['image'][idx]
        lbl_path = self.train_list['label'][idx]

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = encode_labels(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.transform == 'train':
            return apollo_transform(sample)
        else:
            return val_transform(sample)

    def __len__(self):
        return len(self.train_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    data_dir = '../data_list/train.csv'

    apollo_val = ApolloScape(data_dir, '../data/Image_Data', '../data/Gray_Label')

    dataloader = DataLoader(apollo_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_labels(tmp)
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
