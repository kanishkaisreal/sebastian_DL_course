import time 
import numpy as np 
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

device = torch.device('mps')

# set the hyperparameters
random_seed = 123 
learning_rate = 0.0001  
num_epochs = 5 
batch_size  = 128 

# architecture 
num_classes = 10

## MNIST DATASET
# in the code below transforms.ToTensor() scales input  image to 0-1 range 

train_dataset = datasets.MNIST(root= 'data', train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset = test_dataset, 
                         batch_size=batch_size,
                         shuffle=False)


# lets check the dataset
for images, labels in train_loader:
    print('image shape ', images.shape)
    print('labels shape ', labels.shape)
    break 


#RESNET MODEL Architecture 

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        # super(ConvNet, self).__init__
        super().__init__()   # this is preferable 
        # super(ConvNet, self).__init__()  # this is old style 
        # super().__init__(MLP, self)   # will not work 
# super here means that whenver u have a child clas and want to run init from parent class then use super. 
#https://pythonprogramming.net/building-deep-learning-neural-network-pytorch/

        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=4,
                            kernel_size=(1, 1),
                            stride = (1,1),
                            padding = 0),
            torch.nn.BatchNorm2d(4),   # 4 is the out_channels
            torch.nn.ReLU(inplace =True),  # to decrease the memory usage, we use inplace.
            torch.nn.Conv2d(in_channels = 4, 
                            out_channels = 1,
                            kernel_size = (3,3),
                            stride = ( 1, 1),
                            padding = 1),
            torch.nn.BatchNorm2d(1)
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=4,
                            kernel_size=(1, 1),
                            stride = (1,1),
                            padding = 0),
            torch.nn.BatchNorm2d(4),   # 4 is the out_channels
            torch.nn.ReLU(inplace =True),  # to decrease the memory usage, we use inplace.
            torch.nn.Conv2d(in_channels = 4, 
                            out_channels = 1,
                            kernel_size = (3,3),
                            stride = ( 1, 1),
                            padding = 1),
            torch.nn.BatchNorm2d(1)
        )
       # fully connected 
        self.linear_1 =  torch.nn.Linear(1*28*28, num_classes)
       



print('done')