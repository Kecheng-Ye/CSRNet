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


os.environ["CUDA_VISIBLE_DEVICES"] = "3" # GPU ID
# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_data = Data(args.dataset_path, need_transform = args.resample)

    data_loader = DataLoader(train_data,
                             batch_size = args.batch_size, 
                             shuffle = True, 
                             num_workers = args.num_workers) 


    # Define model, Loss, and optimizer
    model = CSRNet(args.configure_path).to(device)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
    criterion = nn.MSELoss(reduction = 'sum')

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()

        # mini-batch
        for i, data in enumerate(data_loader):
            images = data['image'].to(device)
            ground_truth = data['Ground_Truth'].to(device)
            
            # Forward, backward and optimize
            outputs = model(images).double()
            loss = criterion(outputs, ground_truth)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:3f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                print('save model')
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net_' + str(epoch) + '.ckpt'))
        
        # record time for each epoch
        t2 = time.time()
        print(t2 - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models/', help='path for saving trained models')
    parser.add_argument('--configure_path', type = str, default = 'net_configure/configure_3.txt', help = 'path for modeling the models')
    parser.add_argument('--dataset_path', type=str, default = 'data/part_A_final/train_data', help = 'path for reading the data')
    parser.add_argument('--log_step', type = int, default = 10, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 85, help = 'step size for saving trained models')

    # data loader parameters
    parser.add_argument('--resample', type = bool, default = True)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_workers', type = int, default = 4)
    
    # training parameter
    parser.add_argument('--num_epochs', type = int, default = 10)
    parser.add_argument('--learning_rate', type = float, default = 1e-6)
    
    args = parser.parse_args()
    main(args)