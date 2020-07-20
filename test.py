import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data_loader import Data
from network import CSRNet
from utils.evulation_tool import MAE, MSE

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_data = Data(args.dataset_path, need_transform = False)
    data_loader = DataLoader(train_data, 
                             batch_size = 1, 
                             shuffle = False, 
                             num_workers = 4) 
    
    # define load your model here
    model = CSRNet(args.configure_path).to(device)

    PATH = os.path.join(args.model_path, args.model_name)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    factor = checkpoint['factor']
    model.eval()
    
    X = []
    Y = []
    
    # make the model as eval mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images = data['image'].to(device)
            ground_truth = data['Ground_Truth'].to(device)

            output = model(images).cpu().detach().numpy()
            target = ground_truth.cpu().detach().numpy()

            X.append(output)
            Y.append(target)
    
    mae = MAE(X, Y, factor)
    mse = MSE(X, Y, factor)

    record = 'Epoch: {} Traning Loss: {:3f} MAE: {:.1f}  MSE: {:.1f}\n'.format(epoch, loss, mae, mse)
    print(record)
    with open(args.record_file, "a") as f:
        f.write(record)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models/B_set_Adam_factor_2000', help = 'path for saving trained models')
    parser.add_argument('--dataset_path', type = str, default = 'data/part_B_final/test_data')
    parser.add_argument('--configure_path', type = str, default = 'net_configure/configure_3.txt', help = 'path for modeling the models')
    parser.add_argument('--model_name', type = str, default = 'net_19.tar',help = 'model name for the specific model')
    parser.add_argument('--record_file', type = str, default = 'record.txt',help = '')
    parser.add_argument('--number_of_epoch',type = int, default = 0)
    args = parser.parse_args()

    main(args)
