# coding=utf-8

from ctypes.wintypes import tagRECT
import os
import json
import random

from matplotlib import scale
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

# Dataset for FSC147_138
class FSC147_138(Dataset):
    def __init__(self, root, transform, mode='train'):
        self.root = root
        self.anno_file = os.path.join(root, 'annotation_FSC147_384.json')
        self.data_split_file = os.path.join(root, 'Train_Test_Val_FSC_147.json')
        self.im_dir = os.path.join(root, 'images_384_VarV2')
        self.gt_dir = os.path.join(root, 'gt_density_map_adaptive_384_VarV2')
        
        self.transform = transform
        self.mode = mode
        
        with open(self.anno_file) as f:
            annotations = json.load(f)

        with open(self.data_split_file) as f:
            data_split = json.load(f)
            
        self.im_names = data_split[mode]
        self.annotations = {k: annotations[k] for k in self.im_names}
    
    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_path = os.path.join(self.im_dir, im_name)
        anno = self.annotations[im_name]
        
        im = Image.open(im_path).convert('RGB')
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])
        dens = np.load(os.path.join(self.gt_dir, im_name.split(".jpg")[0] + '.npy')).astype('float32')
        
        rects = []
        center_indexes = []
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
            
            center_index = np.where((dots[:,0]>x1) & (dots[:,0]<x2) & (dots[:,1]>y1) & (dots[:,1]<y2))[0]
            if len(center_index) == 0:
                dots = np.append(dots, [[(x1+x2)/2, (y1+y2)/2]], axis=0)
                center_indexes.append(len(dots)-1)
                # print(f"{im_name}")
            elif len(center_index) == 1:
                center_indexes.append(center_index[0])
            else:
                center_p = [(x2+x1)/2, (y2+y1)/2]
                # print(center_p)
                import scipy.spatial
                kdt = scipy.spatial.KDTree(dots[center_index], leafsize=64)
                _, distances_idx = kdt.query(center_p, k=1)
                center_indexes.append(center_index[distances_idx])
        rects = np.array(rects)
        
        ori_im_size = im.size
        if im.size[0] % 8 != 0 or im.size[1] % 8 != 0:
            im = im.resize((int(im.size[0]/8)*8, int(im.size[1]/8)*8), Image.ANTIALIAS)
            dens = cv2.resize(dens, (im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
            if ori_im_size[0] != im.size[0] or ori_im_size[1] != im.size[1]:
                scale_h = im.size[0] / ori_im_size[0]
                scale_w = im.size[1] / ori_im_size[1]
                dots[:,0] = dots[:,0]*scale_w
                dots[:,1] = dots[:,1]*scale_h
                
                rects[:,0] = rects[:,0]*scale_h
                rects[:,1] = rects[:,1]*scale_w
               
               
               
        if self.transform is not None:
            im = self.transform(im)
        
        dens, rects = torch.from_numpy(dens), torch.from_numpy(rects)
        return im, dens, rects, torch.from_numpy(dots).float(), im_name, torch.tensor(center_indexes, dtype=torch.long)
    
    def crop_bboxes(self, img, dens, rects):
        bboxes_dens = []
        bboxes_img = []
        # crop patch for each bbox
        for rect in rects:
            y1 = rect[0]
            x1 = rect[1]
            y2 = rect[2]
            x2 = rect[3]
            # bboxes_dens.append(dens[y1:y2, x1:x2])
            bboxes_img.append(img[:,y1:y2, x1:x2])
        return bboxes_dens, bboxes_img
    
    # random crop augumentation
    def random_crop(self, img, dots, rects, dens, crop_size=384):
        result_img = torch.zeros([3, crop_size, crop_size])
        result_dens = np.zeros([crop_size, crop_size], dtype=np.float32)
        
        content_area_size = crop_size * 7/8.
        
        choice_num = np.random.randint(1, len(rects))
        # randomly select bboxes
        choice_rects = np.array(random.sample(list(rects), choice_num))
        
        left = np.min(choice_rects[:,1])
        right = np.max(choice_rects[:,3])
        top = np.min(choice_rects[:,0])
        bottom = np.max(choice_rects[:,2])
        crop_height, crop_width = bottom - top, right - left
        crop_center_x, crop_center_y = (left + right) / 2, (top + bottom) / 2
        
        crop_max_size = max(crop_height, crop_width)
        scale = 1
        # if crop_height > content_area_size or crop_width > content_area_size:
        if crop_max_size > content_area_size:
            scale = content_area_size / crop_max_size
            # if min_rects_size*scale >= 8:          
            resize = transforms.Resize((int(img.size(1) * scale), int(img.size(2) * scale)))
            img =  resize(img)
            dens = cv2.resize(dens, (int(img.size(2)), int(img.size(1))), interpolation=cv2.INTER_NEAREST)
            crop_center_x, crop_center_y = crop_center_x*scale, crop_center_y*scale
            choice_rects = choice_rects*scale
            rects = rects*scale
            dots = dots*scale
        
        start_y, start_x = max(int(crop_center_y - crop_size/2),0), max(int(crop_center_x - crop_size/2),0)
        end_y, end_x = min(int(crop_center_y + crop_size/2), img.shape[1]), min(int(crop_center_x + crop_size/2), img.shape[2])
        
        # copy the cropped image
        result_img[:, :(end_y-start_y), :(end_x-start_x)] = img[:, start_y:end_y, start_x:end_x]
        result_dens[:(end_y-start_y), :(end_x-start_x)] = dens[start_y:end_y, start_x:end_x]
                
        idx = (dots[:, 0] >= start_x) & (dots[:, 0] <= end_x) & (dots[:, 1] >= start_y) & (dots[:, 1] <= end_y)
        result_dots = dots[idx]
        result_dots[:, 0] -= start_x
        result_dots[:, 1] -= start_y
        
        include_rects = []
        for rect in rects:
            if rect[0] >= start_y and rect[2] <= end_y and rect[1] >= start_x and rect[3] <= end_x:
                rect = np.array([rect[0] - start_y, rect[1] - start_x, rect[2] - start_y, rect[3] - start_x])
                include_rects.append(rect)

        return result_img, torch.from_numpy(result_dots), torch.tensor(include_rects), torch.from_numpy(result_dens)


def collate_fn(batch):
    imgs = []
    dens = []
    dots = []
    im_names = []
    rects = []
    center_indexes = []
    for idx, sample in enumerate(batch):
        imgs.append(sample[0])
        dens.append(sample[1])
        rects.append(sample[2])
        dots.append(sample[3])
        im_names.append(sample[4])
        center_indexes.append(sample[5])
    
    return torch.stack(imgs, 0), torch.stack(dens, 0), rects, dots, im_names, center_indexes
