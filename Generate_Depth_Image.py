import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
# from PIL import Image
import cv2
import os
import time
from api import PRN

import utils.depth_image as DepthImage
import glob
import tqdm
import os
import shutil
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CelebA_live(Dataset):
    def __init__(prn):
        prn.img_list = glob.glob("./cropped_face/live/*.jpg")
        prn.img_size = 256
    
    def __len__(prn):
        return len(prn.img_list)

    def __getitem__(prn, idx):
        img = imread(prn.img_list[idx])
        
        full_name = prn.img_list[idx].split("/")[-1]
        name = full_name.split("_")[0]+".jpg"

        shape = [img.shape[0], img.shape[1]]
        
        str_bbox = full_name.split("_")[1:-1]
        str_bbox.append(full_name.split("_")[-1].replace(".jpg", ""))
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
    prn = PRN(bbox_include=True) 
    shutil.rmtree("./results", ignore_errors=True)
    os.makedirs("./results")
    shutil.rmtree("./comparision", ignore_errors=True)
    os.makedirs("./comparision")

    loader = DataLoader(CelebA_live(), batch_size=64, num_workers=4, collate_fn=collate_fn)
    for item in tqdm.tqdm(loader):
        imgs, names, shapes, bboxes = item
        
        # t0 = time.time()
        cropped_poses, tforms = prn.predict_batch(imgs, shapes, bboxes)
        # t1 = time.time()
        # print("predict batch took {}".format(t1-t0))

        for idx in range(len(cropped_poses)):
            pos = prn.postprocess(cropped_poses[idx], tforms[idx])
            depth_map = prn.create_depth_map(pos, shapes[idx])

            cv2.imwrite("./results/{}".format(names[idx]), depth_map)
            height, width = depth_map.shape
            holder = np.ones((height, width, 3))
            holder[:, :, 0] = depth_map
            holder[:, :, 1] = depth_map
            holder[:, :, 2] = depth_map

            out_img = cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB)
            cv2.imwrite("./comparision/{}".format(names[idx]), cv2.hconcat([out_img, holder.astype(np.uint8)]))

        # t2 = time.time()
        # print("create depth map took {}".format(t2-t1))