import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, Dataset

from data_loader import Data
from network import CSRNet
from torch import tensor


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_data = Data(args.dataset_path, need_transform = args.resample, factor = args.factor)

    data_loader = DataLoader(train_data,
                             batch_size = args.batch_size, 
                             shuffle = True, 
                             num_workers = args.num_workers) 


    # Define model, Loss, and optimizer
    model = CSRNet(args.configure_path).to(device)

    # define optimizer
    # optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
    optimizer = optim.Adam(params = model.parameters(), lr = args.learning_rate)
    criterion = nn.MSELoss(reduction = 'mean')

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()

        # mini-batch
        for i, data in enumerate(data_loader):
            images = data['image'].to(device)
            ground_truth = data['Ground_Truth'].to(device).float()
            
            # Forward, backward and optimize
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:3f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))


        # save model for each epoch
        print('save model')
        PATH = os.path.join(args.model_path, 'net_' + str(epoch) + '.tar')

        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'factor': args.factor
        }, PATH)
                
        
        # record time for each epoch
        t2 = time.time()
        print(t2 - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models/B_set_Adam_factor_2000', help='path for saving trained models')
    parser.add_argument('--configure_path', type = str, default = 'net_configure/configure_3.txt', help = 'path for modeling the models')
    parser.add_argument('--dataset_path', type=str, default = 'data/part_B_final/train_data', help = 'path for reading the data')
    parser.add_argument('--log_step', type = int, default = 50, help = 'step size for prining log info')
    # parser.add_argument('--save_step', type = int, default = 450, help = 'step size for saving trained models')

    # data loader parameters
    parser.add_argument('--resample', type = bool, default = False)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--factor', type = int, default = 2000)
    
    # training parameter
    parser.add_argument('--num_epochs', type = int, default = 20)
    parser.add_argument('--learning_rate', type = float, default = 1e-6)
    
    args = parser.parse_args()
    main(args)


