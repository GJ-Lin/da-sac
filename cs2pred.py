'''
Author: Jay Lin
Date: 2022-12-22 14:22:58
LastEditTime: 2022-12-27 22:08:17
FilePath: /cs2pred.py
'''
import os
import cv2
import shutil
import numpy as np
from collections import namedtuple

src = 'input_multi/dataset'
dst = 'output_multi/03'
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = (
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
)

for dirpath, dirnames, filenames in os.walk(os.path.join(src, 'instance_image')):
    # print(filenames)
    # print(sorted(filenames))
    filenames.sort(key=lambda x: int(x[:-4]))
    # print(filenames)
    # exit(0)
    times_txt = open(os.path.join(dst, 'times.txt'), 'w')
    for i, file in enumerate(filenames):
        img_path = os.path.join(dirpath, file)
        print(img_path)
        # continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print(img.shape)
        pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # print(pred)
        # print(pred.shape)
        for label in labels:
            # print(img[0][0])
            mask = np.all(img == label.color, axis=-1)
            pred[mask] = label.trainId
            # print(mask)
            # print(mask.shape)
            # exit(0)
            # print(img == label.color)
            # print(label.name, label.id, label.trainId, label.color)
            # pred[img == label.color] = label.trainId
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
        times_txt.writelines('%6e\n' % i)
        # exit(0)