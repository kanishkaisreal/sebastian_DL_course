import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys 
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
os.chdir(current_dir)
# # Get the directory where the script is located
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to the CSV file relative to the script directory
# csv_file_path = os.path.join(current_dir, 'mnist_train.csv')



# 1)  inspect the dataset 

df_train = pd.read_csv('mnist_train.csv')
print(df_train.shape)
print(df_train.head())



import torch
from PIL import Image
from torch.utils.data import Dataset
import os

# 2) Custom Dataset Class
class MyDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    

# 3)  Custom Data loader 
    
from torchvision import transforms
from torch.utils.data import DataLoader


# Note that transforms.ToTensor()
# already divides pixels by 255. internally

custom_transform = transforms.Compose([#transforms.Lambda(lambda x: x/255.), # not necessary
                                       transforms.ToTensor()
                                      ])

train_dataset = MyDataset(csv_path='mnist_train.csv',
                          img_dir='mnist_train',
                          transform=custom_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          drop_last=True,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0) # number processes/CPUs to use


valid_dataset = MyDataset(csv_path='mnist_valid.csv',
                          img_dir='mnist_valid',
                          transform=custom_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=100,
                          shuffle=False,
                          num_workers=0)



test_dataset = MyDataset(csv_path='mnist_test.csv',
                         img_dir='mnist_test',
                         transform=custom_transform)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=100,
                         shuffle=False,
                         num_workers=0)

# 4) Iterating Through the Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

num_epochs = 2
for epoch in range(num_epochs):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)

x_image_as_vector = x.view(-1, 28*28)
print(x_image_as_vector.shape)
