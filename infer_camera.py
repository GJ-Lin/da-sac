import os
import sys
import cv2
import time
import numpy as np
import threading
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transform
import torchvision.transforms.functional as tf

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model
from datasets import get_num_classes
from datasets.dataloader_infer import get_dataloader 
from utils.sys_tools import check_dir

from infer_val import convert_dict, mask_overlay


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CamCap:
    def __init__(self, id):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.cap = cv2.VideoCapture(id)

    def start(self):
        print('cam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        print('cam stopped!')

    def getFrame(self):
        return self.Frame
    
    def getStatus(self):
        return self.status

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.cap.read()
            # print(self.status)

class Dasac(object):
    def __init__(self, args):
        # reading the config
        cfg_from_file(args['cfg_file'])
        if args['set_cfgs'] is not None:
            cfg_from_list(args['set_cfgs'])
        num_classes = get_num_classes(args)

        # Loading the model
        self.model = get_model(cfg.MODEL, 0, num_classes=num_classes)
        assert os.path.isfile(args['resume']), "Snapshot not found: {}".format(args['resume'])
        state_dict = convert_dict(torch.load(args['resume'])["model"])
        print(self.model)
        self.model.load_state_dict(state_dict, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False

        # setting the evaluation mode
        self.model.eval()
        # model = nn.DataParallel(model).cuda()
        self.model = nn.DataParallel(self.model).cpu()

        self.infer_dataset = get_dataloader(args['dataloader'], cfg, args['infer_list'])
        self.palette = self.infer_dataset.get_palette()
    
    def infer(self, image):
        image = tf.to_tensor(image)

        imnorm = transform.Normalize(MEAN, STD)
        image = imnorm(image)
        
        # 原来经过 DataLoader 时升高了一维，需要包裹一个 batch_size
        image = image.view(1,*image.size())

        with torch.no_grad():
            _, logits = self.model(image, teacher=False)
            masks_pred = F.softmax(logits, 1)

        image = self.infer_dataset.denorm(image)

        image = image[0]
        masks = masks_pred[0].cpu()
        masks_raw = masks.numpy()
        pred = np.argmax(masks_raw, 0).astype(np.uint8)

        masks = pred
        images = image.numpy()
        images = np.transpose(images, [1,2,0])

        overlay = mask_overlay(masks, images, self.palette)
        img = cv2.cvtColor(np.asarray((overlay * 255.).astype(np.uint8)), cv2.COLOR_RGB2BGR)
        return img

if __name__ == '__main__':
    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    num_classes = get_num_classes(args)

    # Loading the model
    model = get_model(cfg.MODEL, 0, num_classes=num_classes)
    assert os.path.isfile(args.resume), "Snapshot not found: {}".format(args.resume)
    state_dict = convert_dict(torch.load(args.resume)["model"])
    print(model)
    model.load_state_dict(state_dict, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    # setting the evaluation mode
    model.eval()
    # model = nn.DataParallel(model).cuda()
    model = nn.DataParallel(model).cpu()

    infer_dataset = get_dataloader(args.dataloader, cfg, args.infer_list)
    palette = infer_dataset.get_palette()

    cap = CamCap(0)
    cap.start()
    time.sleep(1)
    while(True):
        frame = cap.getFrame()
        if not cap.getStatus():
            print("Failed to read the image.")
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = tf.to_tensor(image)

        imnorm = transform.Normalize(MEAN, STD)
        image = imnorm(image)
        
        # 原来经过 DataLoader 时升高了一维，需要包裹一个 batch_size
        image = image.view(1,*image.size())

        with torch.no_grad():
            _, logits = model(image, teacher=False)
            masks_pred = F.softmax(logits, 1)

        image = infer_dataset.denorm(image)

        image = image[0]
        masks = masks_pred[0].cpu()
        masks_raw = masks.numpy()
        pred = np.argmax(masks_raw, 0).astype(np.uint8)

        masks = pred
        images = image.numpy()
        images = np.transpose(images, [1,2,0])

        overlay = mask_overlay(masks, images, palette)
        img = cv2.cvtColor(np.asarray((overlay * 255.).astype(np.uint8)), cv2.COLOR_RGB2BGR)

        cv2.imshow('Video', img)
        key = cv2.waitKey(1)
        # press Esc to quit
        if key == 27:
            break
    
    cap.stop()
    cv2.destroyAllWindows()
