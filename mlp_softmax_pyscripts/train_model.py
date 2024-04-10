# imports from helper.py
from helper import get_dataloaders_mnist, set_all_seeds, set_deterministic
from helper import compute_accuracy, plot_training_loss, plot_accuracy

# standard library
import argparse
import logging
import os
import time
import PIL 

# installed libraries
import torch
from torchvision import transforms
import yaml  # conda install pyyaml

torch.use_deterministic_algorithms(True)  # Enable deterministic algorithms

parser = argparse.ArgumentParser()
parser.add_argument('--settings_path',
                    type=str,
                    required=True)
parser.add_argument('--results_path',
                    type=str,
                    required=True)

settings_path  = '/Users/kanishka/Library/CloudStorage/OneDrive-UTArlington/online_courses/sebastian_Introduction to Deep Learning_course/mlp_softmax_pyscripts/settings.yaml'
results_path = '/Users/kanishka/Library/CloudStorage/OneDrive-UTArlington/online_courses/sebastian_Introduction to Deep Learning_course/mlp_softmax_pyscripts/results'

if not os.path.exists(results_path):
    os.makedirs(results_path)
with open(settings_path) as file:
    SETTINGS = yaml.load(file, Loader=yaml.FullLoader)

# args = parser.parse_args()
# if not os.path.exists(args.results_path):
#     os.makedirs(args.results_path)
# with open(args.settings_path) as file:
#     SETTINGS = yaml.load(file, Loader=yaml.FullLoader)

# this is a way to record all the output.
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logpath = os.path.join(results_path, 'training.log')
# logpath = os.path.join(args.results_path, 'training.log')
logger.addHandler(logging.FileHandler(logpath, 'a'))
print = logger.info


print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    device = torch.device(f'cuda:{SETTINGS["cuda device"]}')
else:
    device = torch.device('cpu')
print(f'Using {device}')

set_all_seeds(SETTINGS['random seed'])
set_deterministic()

training_transforms = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.RandomCrop(size=(28, 28)),
    transforms.RandomRotation(degrees=30, interpolation=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
    # normalize does (x_i - mean) / std
    # if images are [0, 1], they will be [-1, 1] afterwards
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(32, 32)),
    transforms.CenterCrop(size=(28, 28)),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])


train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    train_transforms=training_transforms, # transforms.ToTensor(), training_transforms
    test_transforms=test_transforms, # transforms.ToTensor(), test_transforms
    batch_size=SETTINGS['batch size'],
    num_workers=SETTINGS['num workers'],
    validation_fraction=SETTINGS['validation fraction'])


##########################
# ## MODEL
##########################

class MLP(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()
        # super().__init__()   # this is preferable 
        # you can also write 
        # super(MLP, self).__init__()  # this is old style 
        # super().__init__(MLP, self)   # will not work 
# super here means that whenver u have a child clas and want to run init from parent class then use super. 
#https://pythonprogramming.net/building-deep-learning-neural-network-pytorch/
        self.num_classes = num_classes

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, num_hidden),  # Hidden Layer
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_hidden, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.classifier(x)


model = MLP(num_features=SETTINGS['input size'],
            num_hidden=SETTINGS['hidden layer size'],
            num_classes=SETTINGS['num class labels'])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning rate'])


start_time = time.time()
minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
for epoch in range(SETTINGS['num epochs']):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(device)
        targets = targets.to(device)

        # ## FORWARD AND BACK PROP
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()

        loss.backward()

        # ## UPDATE MODEL PARAMETERS
        optimizer.step()

        # ## LOGGING
        minibatch_loss_list.append(loss.item())
        if not batch_idx % 50:
            print(f'Epoch: {epoch+1:03d}/{SETTINGS["num epochs"]:03d} '
                  f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                  f'| Loss: {loss:.4f}')

    model.eval()
    with torch.no_grad():  # save memory during inference
        train_acc = compute_accuracy(model, train_loader, device=device)
        valid_acc = compute_accuracy(model, valid_loader, device=device)
        print(f'Epoch: {epoch+1:03d}/{SETTINGS["num epochs"]:03d} '
              f'| Train: {train_acc :.2f}% '
              f'| Validation: {valid_acc :.2f}%')
        train_acc_list.append(train_acc.item())
        valid_acc_list.append(valid_acc.item())

    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

test_acc = compute_accuracy(model, test_loader, device=device)
print(f'Test accuracy {test_acc :.2f}%')

# ######### MAKE PLOTS ######
plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=SETTINGS['num epochs'],
                   iter_per_epoch=len(train_loader),
                   results_dir=args.results_path)
plot_accuracy(train_acc_list=train_acc_list,
              valid_acc_list=valid_acc_list,
              results_dir=args.results_path)

results_dict = {'train accuracies': train_acc_list,
                'validation accuracies': valid_acc_list,
                'test accuracy': test_acc.item(),
                'settings': SETTINGS}

results_path = os.path.join(args.results_path, 'results_dict.yaml')
with open(results_path, 'w') as file:
    yaml.dump(results_dict, file)