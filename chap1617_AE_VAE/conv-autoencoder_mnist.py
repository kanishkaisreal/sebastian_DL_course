import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

import sys, os 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# import utility functions 
from custom_dataloader.helper_data import get_dataloaders_mnist
from custom_dataloader.helper_train import train_autoencoder_v1
from custom_dataloader.helper_utils import set_deterministic, set_all_seeds
from custom_dataloader.helper_plotting import plot_training_loss
from custom_dataloader.helper_plotting import plot_generated_images
from custom_dataloader.helper_plotting import plot_latent_space_with_labels

# device settings 

print('done')