# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2020-01-23 17:43
# @Author   : Fabrice LI
# @File     : tensorboard_summary.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************
import os
import numpy as np
import torch
import cv2

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from utils.dataset import Apolloscapes, decode_color_labels


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def get_color_mask(self, pred):
        pred = pred.numpy()
        pred = np.array(pred[0]).astype(np.uint8)
        pred = decode_color_labels(pred)

        return torch.from_numpy(pred)

    def visualize_image(self, writer, image, ground_truth, output_model, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        
        grid_ground_truth = make_grid(self.get_color_mask(ground_truth.cpu()))
        writer.add_image('Ground truth', grid_ground_truth, global_step)

        output_model_color = self.get_color_mask(torch.max(output_model[:3], 1)[1].cpu())
        grid_output_model = make_grid(output_model_color, 3, normalize=False,
                                      range=(0, 255))
        writer.add_image('Predicted', grid_output_model, global_step)

        fusion_image = self.fusion(image.cpu(), output_model_color.cpu())
        grid_fusion_image = make_grid(fusion_image, 3, normalize=False, range=(0, 255))
        writer.add_image('Fusion', grid_fusion_image, global_step)

    # image output_model: np.array
    def fusion(self, image, output_model):
        image = image.detach().numpy()
        image = np.array(image[0]).astype(np.uint8)
        output_model = output_model.detach().numpy()
        image_add = cv2.addWeighted(image, alpha=0.5, src2=output_model, beta=0.5, gamma=0)
        return torch.from_numpy(image_add)


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader

    summary = TensorboardSummary(directory='../logs')
    writer = summary.create_summary()
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_dataset = Apolloscapes('../data/train_dataset.csv', '../data/Image_Data', 'data/Gray_Label',
                                      [1024, 386], type='train')

    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, **kwargs)
    data = tqdm(dataloader)
    length = len(dataloader)
    for i, sample in enumerate(data):
        image, label = sample['image'], sample['label']
        summary.visualize_image(writer, image, label, label, i)
        # img = sample['image'].numpy()
        # s = summary.get_color_mask(label)
        # gt = label.numpy()
        # tmp = np.array(gt[0]).astype(np.uint8)
        # segmap = decode_color_labels(tmp)
        # img_tmp = np.transpose(img[0], axes=[1, 2, 0])
        # img_tmp = img_tmp.astype(np.uint8)
        # segmap_t = np.transpose(segmap, axes=[1, 2, 0])
        # segmap_t = segmap_t.astype(np.uint8)
        # plt.figure()
        # plt.title('display')
        # plt.subplot(211)
        # plt.imshow(img_tmp)
        # plt.subplot(212)
        # plt.imshow(segmap_t)
        # label = torch.from_numpy(segmap)

        if i == 5:
            break
    plt.show(block=True)
