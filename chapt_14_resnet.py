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
            torch.nn.Conv2d(in_channels=1,  # this in_channels =1 snce MNIST is 1 channel image so input is 1 channel only. 
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
        
        # in each block 1 and block 2 we are doing rom 1 to 4 channel and then back from 4 to 1 channel. 
       # fully connected 
        self.linear_1 =  torch.nn.Linear(1*28*28, num_classes)
    
    def forward(self,x ):
        # 1 residual block 
        shortcut = x 
        x = self.block_1(x)
        x = torch.nn.functional.relu(x + shortcut)
        
        # 2 residual block 
        shortcut = x 
        x = self.block_2(x)
        x = torch.nn.functional.relu(x + shortcut)
        
        # fully connected 
        logits = self.linear_1(x.view(-1, 1 * 28*28))
        return logits

torch.manual_seed(random_seed)
model = ConvNet(num_classes= num_classes)
model = model.to(device)

optimizer  = torch.optim.Adam(model.parameters(), lr = learning_rate)

# now we do the training 

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):            
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

start_time = time.time()
for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass 
        logits = model(features)
        cost = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        # backward pass
        cost.backward()
        
        # update model parameters 
        optimizer.step()
        
        # log the model training details 
        if not batch_idx % 250:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost : %0.4f' 
                  %(epoch+1, num_epochs, batch_idx,len(train_loader), cost))
    
    model = model.eval() # this will prevent updating batchnorm parameters during inference 
    with torch.set_grad_enabled(False): # this is to save memory during inference
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))        
    
    

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__
        
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels[0],
                            out_channels=channels[1],
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1),
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.conv2d(in_channels=channels[1],
                            out_channels = channels[2],
                            kernel_size = ( 1,1 ), 
                            stride = ( 1,1 ),
                            padding = 0 ),
            torch.nn.BatchNorm2d(channels[2])
        )        
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels[0],
                            out_channels=channels[2],
                            kernel_size=(1,1),
                            stride=(2,2),
                            padding=0),
            torch.nn.BatchNorm2d(channels[2])
        )
    
    def forward(self, x):
        shortcut = x 
        block = self.block(x)
        shortcut = self.shortcut(x)
        x = torch.nn.functional.relu(block+ shortcut)
        return x 
    
    
                

print('done')