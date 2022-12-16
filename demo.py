'''
Author: Jay Lin
Date: 2022-12-16 21:55:52
LastEditTime: 2022-12-16 22:26:41
FilePath: /demo.py
'''

import cv2
from infer_camera import Dasac
from PIL import Image

if __name__ == '__main__':
    args = {
        'cfg_file': 'configs/deeplabv2_resnet101_train.yaml',
        'set_cfgs': [],
        'resume': 'snapshots/cityscapes/baselines/resnet101_gta/final_e136.pth',
        'dataloader': 'cityscapes',
        'infer_list': 'val_cityscapes'
    }
    dasac = Dasac(args)
    image = Image.open('input/test_input.png').convert('RGB')
    img_infer = dasac.infer(image)
    cv2.imshow('img_infer', img_infer)
    cv2.waitKey(0)