from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageEnhance
import torch
import pandas as pd
from collections import namedtuple
import pdb

class KaggleBrainMRIDataset(Dataset):

    def __init__(self, csv_file, config_data, mode):
        self.data      = pd.read_csv(csv_file, header=None)
        self.CropTransform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        self.NormTransform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.n_class = 2
        self.transformList = []
        if(mode == 'train'):
            #Don't apply transformations for validation and test images
            if(config_data['transforms']['RotateAngle'] != 0):
                self.transformList.append('Rotate')
                self.RotateAngle = config_data['transforms']['RotateAngle']
            if(config_data['transforms']['HorzFlip'] == 'true'):
                self.transformList.append('Flip')
            if(config_data['transforms']['BrightnessFactor'] != 1):
                self.transformList.append('Brighten')
                self.BrightnessFactor = config_data['transforms']['BrightnessFactor']
            if(config_data['transforms']['ContrastFactor'] != 1):
                self.transformList.append('Contrast')
                self.ContrastFactor = config_data['transforms']['ContrastFactor']
            if(config_data['transforms']['SharpnessFactor'] != 1):
                self.transformList.append('Sharpen')
                self.SharpnessFactor = config_data['transforms']['SharpnessFactor']
        #print(self.data.shape)
    
    def FlipRotate(self, image, angle):
        randFlip = np.random.random()
        if(randFlip > 0.5):
            image = transforms.functional.hflip(image)
        #Do rotation now
        angle = transforms.RandomRotation(angle).get_params([-1*angle , angle])
        image = transforms.functional.rotate(image, angle)
        return image

    def Rotate(self, image, angle):
        angle = transforms.RandomRotation(angle).get_params([-1*angle , angle])
        image = transforms.functional.rotate(image, angle)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #pdb.set_trace()
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert('RGB')
        # pdb.set_trace()
        #Apply the transforms
        for transform in self.transformList:
            if(transform == 'Flip'):
                randFlip = np.random.random()
                if(randFlip > 0.5):
                    img = transforms.functional.hflip(img)
            if(transform == 'Rotate'):
                img = self.Rotate(img, self.RotateAngle)
            if(transform == 'Brighten'):
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(self.BrightnessFactor)
            if(transform == 'Contrast'):
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.ContrastFactor)
            if(transform == 'Sharpness'):
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(self.SharpnessFactor)

        img = self.CropTransform(img)
        label = self.data.iloc[idx, 1]
        img = np.asarray(img) / 255.# scaling [0-255] values to [0-1]
        img = self.NormTransform(img)
        #img = img.permute([2,0,1])
        target = torch.zeros(self.n_class,)
        target[label] = 1
            
        return img, target, label
