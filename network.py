import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torch.nn.modules.container import Sequential


class CSRNet(nn.Module):
    def __init__(self, configure_path : str):
        super(CSRNet, self).__init__()

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    
        self.front_end = vgg16(pretrained = True).features[:23]
        out_channels = self.front_end[-2].out_channels

        self.back_end = self.parse_configure(configure_path, out_channels)
        self.back_end.apply(self.initialization)

        self.upsample = nn.Upsample(scale_factor = 8, mode = 'bilinear')

    
    def forward(self, x):
        result = self.front_end(x)
        result = self.back_end(result)
        result = self.upsample(result)
        return result
        

    def parse_configure(self, configure_path : str, in_channels : int):
        if not os.path.exists(configure_path):
            print('Wrong configure path file')
            pass
        
        model_list = []
        config_file = open(configure_path, 'r') 

        # each line means a new conv layer
        for line in config_file.readlines(): 
            line = line[4 : ]
            # seperate each parameter
            kernel_size, out_channels, dilation_rate = tuple(int(parm) for parm in line.split('-'))
            padding_size = int((kernel_size - 1 + (kernel_size - 1) * (dilation_rate - 1))/2)
            # add the conv layer and relu to the list
            model_list.append(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, dilation = dilation_rate, padding = padding_size))
            model_list.append(nn.ReLU(inplace = True))
            
            in_channels = out_channels
        
        # the last layer, we want the image to be single channel
        model_list.append(nn.Conv2d(in_channels, 1, kernel_size = 1, dilation = 1))
        model_list.append(nn.ReLU(inplace = True))

        return nn.Sequential(*model_list)


    def initialization(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std = 0.01)
            nn.init.normal_(m.bias.data, std = 0.01)
        

    

# CSRNet = CSRNet('net_configure/configure_3.txt')
# print(CSRNet.parameters())
# _input = torch.randn((1, 3, 768, 1024))
# print(CSRNet.front_end(_input).size())
# _input = torch.randn((1, 512, 96, 128))
# print(CSRNet.back_end(_input).size())
# _input = torch.randn((1, 1, 60, 92))


# _input = torch.randn((1, 3, 1024, 768))
# print(CSRNet.front_end(_input).size())
# _input = torch.randn((1, 512, 128, 96))
# print(CSRNet.back_end(_input).size())
# _input = torch.randn((1, 1, 92, 60))
# print(CSRNet.upsample(_input).size())

