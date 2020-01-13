# -*-coding:utf-8 -*-
#Reference:**********************************************
# @Time     : 2020-01-11 23:44
# @Author   : Fabrice LI
# @File     : predict.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
#Reference:**********************************************
import os
import torch
from PIL import Image
import numpy as np

from Config import Config
from model.deeplab import DeepLab
from utils.utils import decode_color_labels

os.environ['CUDA_VISIBLE_DEVICES'] = ''

device_id = 0

predict_model = "deeplabv3plus"

models = {"deeplabv3plus": DeepLab}


def load_model(model_path):
    lande_config = Config()
    model = models[predict_model](backbone=lande_config.BACKBONE,
                                  output_stride=lande_config.OUTPUT_STRIDE,
                                  num_classes=lande_config.NUM_CLASSES,
                                  sync_bn=False, freeze_bn=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=device_id)
        map_location = 'cuda:%d' % device_id
    else:
        map_location = "cpu"
    model_param = torch.load(model_path, map_location=map_location)['state_dict']
    model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
    model.load_state_dict(model_param)
    return model


def crop_resize_data(image, label=None, image_size=(1024, 384), offset=690):
    """
    Attention:
    h,w, c = image.shape
    cv2.resize(image,(w,h))
    """
    if label is not None:
        img_w, img_h = image.size
        mask_w, mask_h = label.size
        img = image.crop((0, offset, img_w, img_h))
        mask = label.crop((0, offset, mask_w, mask_h))

        img = img.resize(image_size, Image.BILINEAR)
        mask = mask.resize(image_size, Image.NEAREST)
        return img, mask
    else:
        img_w, img_h = image.size
        img = image.crop((0, offset, img_w, img_h))
        train_image = img.resize(image_size, Image.BILINEAR)
        return train_image


def img_transform(img):
    img = crop_resize_data(img)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img


def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))

    return pred


def main():
    test_dir = "logs"
    model_path = os.path.join(test_dir, 'finalNet.pth.tar')
    print("Loading model...")
    model = load_model(model_path)
    print("load finished")

    image_path = os.path.join(test_dir, 'test.jpg')
    img = Image.open(image_path).convert('RGB')
    img = img_transform(img)

    print("model inferring...")
    pred = model(img)
    color_mask = get_color_mask(pred)
    result = Image.fromarray((color_mask * 255).astype(np.uint8))
    result.save('mask.jpg')
    print("saved")


if __name__ == '__main__':
    main()
