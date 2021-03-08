################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import pdb
from torchvision import models
import torch.nn as nn
import torch
import ssl

# Build and return the model here based on the configuration.
def get_model(pretrained_model_type):

    ssl._create_default_https_context = ssl._create_unverified_context

    # You may add more parameters if you want
    #newModel = models.resnet50(pretrained=True)
    #Freeze weights
    #for param in newModel.parameters():
    #    param.requires_grad = False
    #Make the new RNN encoder

    #newModel.fc = torch.nn.Linear(in_features=2048, out_features=hidden_size, bias=True)
    if(pretrained_model_type == 'vgg16'):
        return VGG16Model()
    elif(pretrained_model_type == 'resnet50'):
        return ResNet50Model()
    elif(pretrained_model_type == 'densenet121'):
        return DenseNet121Model()
        
class VGG16Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        tempModel = models.vgg16_bn(pretrained=True)
        self.vggFeatureEncoder = nn.Sequential(*(list(tempModel.children())[:-1]))
        self.vggClassifier = nn.Sequential(*(list(tempModel.children())[2][:-1]))
        for param in self.vggFeatureEncoder.parameters():
            param.requires_grad = False
        for param in self.vggClassifier.parameters():
            param.requires_grad = False
        del tempModel
        self.relu = nn.ReLU(inplace=True)
        self.fcLayer = nn.Linear(4096, 2)

    def forward(self, image_in):
        #pdb.set_trace()
        x = self.vggFeatureEncoder(image_in)
        x = self.vggClassifier(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.fcLayer(x)
        return x

class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        tempModel = models.resnet50(pretrained=True)
        self.featureEncoder = nn.Sequential(*(list(tempModel.children())[:-1]))
        for param in self.featureEncoder.parameters():
            param.requires_grad = False
        del tempModel
        self.relu = nn.ReLU(inplace=True)
        self.fcLayer = nn.Linear(2048, 2)

    def forward(self, image_in):
        #pdb.set_trace()
        x = self.featureEncoder(image_in)
        x = self.relu(x.squeeze())
        x = self.fcLayer(x)
        return x

class DenseNet121Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        tempModel = models.densenet121(pretrained=True)
        self.featureEncoder = nn.Sequential(*(list(tempModel.children())[:-1]))
        for param in self.featureEncoder.parameters():
            param.requires_grad = False
        del tempModel
        self.relu = nn.ReLU(inplace=True)
        self.fcLayer = nn.Linear(1024*4*4, 2)

    def forward(self, image_in):
        #pdb.set_trace()
        x = self.featureEncoder(image_in)
        x = self.relu(x.view(x.shape[0], -1))
        x = self.fcLayer(x)
        return x