from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple
import pdb

class KaggleBrainMRIDataset(Dataset):

    def __init__(self, csv_file):
        self.data      = pd.read_csv(csv_file, header=None)
        self.CropTransform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        self.NormTransform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.n_class = 2
        #print(self.data.shape)
    
    def FlipRotate(self, image):
        randFlip = np.random.random()
        if(randFlip > 0.5):
            image = transforms.functional.hflip(image)
        #Do rotation now
        angle = transforms.RandomRotation(45).get_params([-45. , 45.])
        image = transforms.functional.rotate(image, angle)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #pdb.set_trace()
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert('RGB')
        # pdb.set_trace()
        img = self.CropTransform(img)
        label = self.data.iloc[idx, 1]
        
        randNum = np.random.random()
        if(randNum > 0.5):
            img = self.FlipRotate(img)

        img = np.asarray(img) / 255.# scaling [0-255] values to [0-1]
        img = self.NormTransform(img)
        #img = img.permute([2,0,1])
        target = torch.zeros(self.n_class,)
        target[label] = 1
            
        return img, target, label
