import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from skimage import io, transform
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from numpy.lib.shape_base import expand_dims
import torchvision.transforms.functional as F


class Data(Dataset):
    def __init__(self, foldername, need_transform):
        # parameters
        self.img_data_path = '{}/images'.format(foldername)
        self.img_data_path = [os.path.join(self.img_data_path, f)
                              for f in os.listdir(self.img_data_path)
                              if os.path.isfile(os.path.join(self.img_data_path, f))] * 9
        self.ground_truth_path = foldername + '/ground_truth/np_file/GT_{}.npy'
        self.len = len(self.img_data_path)
        self.crop_size = (128, 128)
        self.need_transform = need_transform
    

    def __len__(self):
        return self.len
    

    def __getitem__(self, idx : int):
        img_name = self.img_data_path[idx]
        gt_name = self.ground_truth_path.format(img_name[img_name.find('IMG_') + 4 : -4])
        # print(img_name, gt_name)

        img = cv2.imread(img_name)
        with open(gt_name, 'rb') as f:
            ground_truth = np.load(f)
        
        # ground_truth = expand_dims(ground_truth, axis=2)
        if self.need_transform:
            sample = self.transform(img, ground_truth)
        else:
            sample = {'image': F.to_tensor(img), 
                      'Ground_Truth': F.to_tensor(ground_truth)}

        return sample

    def transform(self, image, ground_truth):
        # get the specific coordinate of cropping
        orginal_shape = image.shape
        orginal_truth = ground_truth
        image = F.to_pil_image(image)
        self.crop_indices = RandomCrop.get_params(
                            image, output_size = self.crop_size)
        i, j, h, w = self.crop_indices
        # crop the image and the GT
        image = F.crop(image, i, j, h, w)
        ground_truth = ground_truth[i : i  + h , j : j + w]
        # send them to tensor
        sample = {'image': F.to_tensor(image), 
                  'Ground_Truth': F.to_tensor(ground_truth)}
        
        return sample




if __name__ == '__main__':
    data = Data('data/part_A_final/train_data', need_transform = True)
    data_loader = DataLoader(data, batch_size = 32, num_workers = 4)

    total_step = len(data_loader)
    for i, data in enumerate(data_loader):
        # mini-batch
        print('[{}/{}]: img shape : {}, GT shape : {}'.format(i, total_step, data['image'].size(), data['Ground_Truth'].size()))
    
