'''
Author: Jay Lin
Date: 2022-12-22 14:22:58
LastEditTime: 2023-03-30 21:49:28
FilePath: /rename_dataset.py
'''
import os
import cv2
import shutil
import numpy as np
# from collections import namedtuple

src = 'input_dataset'
dst = 'output_multi/02'

for dirpath, dirnames, filenames in os.walk(os.path.join(src, 'image')):
    # print(filenames)
    # print(sorted(filenames))
    filenames.sort(key=lambda x: int(x[:-4]))
    # print(filenames)
    # exit(0)
    times_txt = open(os.path.join(src, 'times.txt'), 'w')
    for i, file in enumerate(filenames):
        img_path = os.path.join(dirpath, file)
        print(img_path)
        # os.popen('mv %s %s' % (img_path, os.path.join(src, 'image', '%06d.png' % i)))
        times_txt.writelines('%6e\n' % i)
        # continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print(img.shape)
        pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # print(pred)
        # print(pred.shape)

        print(pred)
        filename = '%06d.png' % (i)
        print(filename)
        # continue
        img_raw = cv2.imread(os.path.join(src, 'image', file[:-4]+'.png'), cv2.IMREAD_UNCHANGED)
        img_raw = cv2.resize(img_raw, (img.shape[1]//2, img.shape[0]//2))
        cv2.imwrite(os.path.join(dst, 'image_2', filename), img_raw)
        # shutil.copyfile(os.path.join(src, 'image', file[:-4]+'.png'), os.path.join(dst, 'image_2', filename))
        # cv2.imwrite(os.path.join(dst, 'image_2', filename), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred = cv2.resize(pred, (img.shape[1]//2, img.shape[0]//2))
        cv2.imwrite(os.path.join(dst, 'seg_result', filename), pred)
        print('%06e' % i)
        # exit(0)