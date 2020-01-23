# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-23 16:01
# @Author   : Fabrice LI
# @File     : make_csv.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import os
import pandas as pd
from sklearn.utils import shuffle


def make_csv_file():
    label_list = []
    image_list = []

    # image_dir = '/root/private/baidu-lane-dect-pytorch/data/Image_Data/'
    # label_dir = '/root/private/baidu-lane-dect-pytorch/data/Gray_Label/'
    image_dir = '/home/aistudio/data/data1919/Image_Data/'
    label_dir = '/home/aistudio/data/data1919/Gray_Label/'

    for s1 in os.listdir(image_dir):
        if s1 == '.DS_Store':
            break
        image_sub_dir1 = os.path.join(image_dir, s1)
        label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')
        # print(s1, image_sub_dir1, label_sub_dir1)

        for s2 in os.listdir(image_sub_dir1):
            if s2 == '.DS_Store':
                break
            image_sub_dir2 = os.path.join(image_sub_dir1, s2)
            label_sub_dir2 = os.path.join(label_sub_dir1, s2)
            # print(image_sub_dir2, label_sub_dir2)

            for s3 in os.listdir(image_sub_dir2):
                if s3 == '.DS_Store':
                    break
                image_sub_dir3 = os.path.join(image_sub_dir2, s3)
                label_sub_dir3 = os.path.join(label_sub_dir2, s3)
                # print(image_sub_dir3, label_sub_dir3)

                for s4 in os.listdir(image_sub_dir3):
                    if s4 == '.DS_Store':
                        break
                    s44 = s4.replace('.jpg', '_bin.png')
                    image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                    label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                    if not os.path.exists(image_sub_dir4):
                        print(image_sub_dir4)
                    if not os.path.exists(label_sub_dir4):
                        print(label_sub_dir4)
                    # print(image_sub_dir4, label_sub_dir4)
                    image_list.append(image_sub_dir4)
                    label_list.append(label_sub_dir4)
    print(len(image_list), len(label_list))

    save = pd.DataFrame({'image': image_list, 'label': label_list})
    save_shuffle = shuffle(save)
    length = len(save_shuffle)
    train_dataset = save_shuffle[: int(length * 0.8)]
    # val_dataset = save_shuffle[int(length * 0.6): int(length * 0.8)]
    test_dataset = save_shuffle[int(length * 0.8):]
    train_dataset.to_csv('/home/aistudio/work/auto-car/train_dataset.csv', index=False)
    # train_dataset.to_csv('/root/private/baidu-lane-dect-pytorch/data_list/train_dataset.csv', index=False)
    test_dataset.to_csv('/home/aistudio/work/auto-car/val_dataset.csv', index=False)
    # test_dataset.to_csv('/root/private/baidu-lane-dect-pytorch/data_list/test_dataset.csv', index=False)


if __name__ == '__main__':
    make_csv_file()
