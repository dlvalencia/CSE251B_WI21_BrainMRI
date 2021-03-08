from torchvision import utils
from torchvision import utils
from dataloader import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import torch.nn as nn
import pdb
import logging
import os
import sys

#pdb.set_trace()
train_dataset = KaggleBrainMRIDataset(csv_file='train_images.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size= 8, num_workers= 0, shuffle=False)
for iter, (images, onehotTarget, label) in enumerate(train_loader):
    #pdb.set_trace()
    inputs = images
    outputs = label
    print(inputs.shape)
    print(outputs.shape)

