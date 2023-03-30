'''
Author: Jay Lin
Date: 2022-12-17 22:26:02
LastEditTime: 2022-12-27 15:54:53
FilePath: /demo_multi.py
'''

import os
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
    # image = Image.open('input/test_input.png').convert('RGB')
    dir_input = 'input_multi'
    dir_output = 'output_multi'
    for dirpath, dirnames, filename in os.walk('input_multi'):
        for file in filename:
            print(os.path.join(dirpath, file))
            image = Image.open(os.path.join(dirpath, file)).convert('RGB')
            img_infer, img_pred = dasac.infer(image, ret_pred=True)
            print(img_infer)
            print(img_pred)
            print(img_infer.shape, img_pred.shape)
            # exit(0)
            cv2.imwrite(os.path.join(dir_output, 'infer', file), img_infer)
            cv2.imwrite(os.path.join(dir_output, 'pred', file), img_pred)
            