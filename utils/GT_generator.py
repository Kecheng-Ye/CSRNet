import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class data_generator():
    def __init__(self, foldername : str):
        self.img_data_path = '{}/images'.format(foldername)
        self.ground_truth_path = '{}/ground_truth'.format(foldername)
        self.len = len([f for f in os.listdir(self.img_data_path)
                        if os.path.isfile(os.path.join(self.img_data_path, f))])
        
        self.generate_GT()

    def generate_GT(self):
        for i in range(1, self.len + 1):
            img_name = os.path.join(self.img_data_path,
                                'IMG_{}.jpg'.format(i))
            gt_name = os.path.join(self.ground_truth_path,
                                'GT_IMG_{}.mat'.format(i))
            
            with open('{}/np_file/GT_{}.npy'.format(self.ground_truth_path, i), 'wb') as f:
                np.save(f, self.ground_truth_generator(loadmat(gt_name), cv2.imread(img_name)))
                print('finish {} image'.format(i))

    def ground_truth_generator(self, mat_data : dict, image : np.ndarray):
        # read the coordinate of each head's in the mat file
        heads = mat_data['image_info'][0][0][0][0][0]
        result = np.zeros(shape = (image.shape[0], image.shape[1]))

        # hyper parameter 
        beta = 0.3
        k = 3
        size = 15

        # generate the density distribution for each head
        for each_head in heads:
            # eliminate deprecated data
            if each_head[1] > image.shape[0] or each_head[1] < 0 :
                continue
            
            if each_head[0] > image.shape[1] or each_head[0] < 0 :
                continue

            # find the k's nearst neighbour
            dist = np.sort(np.linalg.norm(heads - each_head, ord = 2, axis = 1))[1 : 1 + k]
            d_i = np.mean(dist)
            delta_matrix = np.zeros(shape = (image.shape[0], image.shape[1]))
            delta_matrix[int(each_head[1])][int(each_head[0])] = 1
            # Gaussian Blur the one hot head matrix and add to the total density map
            result += cv2.GaussianBlur(delta_matrix, (size, size), beta * d_i)

        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, help='path for target data')
    args = parser.parse_args()
    data_generator = data_generator(args.data_path)
