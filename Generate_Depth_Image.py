import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
# from PIL import Image
import cv2
import os

from api import PRN

import utils.depth_image as DepthImage
import glob
import tqdm
import os
import shutil
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CelebA_live(Dataset):
    def __init__(self):
        self.img_list = glob.glob("./cropped_face/live/*.jpg")
        self.img_size = 256
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = imread(self.img_list[idx])
        
        full_name = self.img_list[idx].split("/")[-1]
        name = full_name.split(".")[0]+".jpg"

        shape = [img.shape[0], img.shape[1]]
        
        str_bbox = full_name.split(".")[1:-2]
        bbox = [float(str_bbox[i]) for i in range(4)]

        return img, name, shape, bbox

def collate_fn(batch):
    imgs = []
    names = []
    shapes = []
    bboxes = []
    for item in batch:
        imgs.append(item[0])
        names.append(item[1])
        shapes.append(item[2])
        bboxes.append(item[3])

    return [imgs, names, shapes, bboxes]

if __name__=="__main__":
    prn = PRN(is_dlib = False, is_opencv = False) 
    shutil.rmtree("./results", ignore_errors=True)
    os.makedirs("./results")

    loader = DataLoader(CelebA_live(), batch_size=2, num_workers=4, collate_fn=collate_fn)
    for item in tqdm.tqdm(loader):
        imgs, names, shape, bboxes = item
        depth_maps = prn.predict_batch(imgs, shape, bboxes)

        for idx in range(len(names)):
            cv2.imwrite("./results/{}".format(names[idx]), depth_maps[idx])

