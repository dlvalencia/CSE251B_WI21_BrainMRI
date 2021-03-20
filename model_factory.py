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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 2
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
    elif(pretrained_model_type == 'FCModel'):
        return Fullyconneted()
    elif(pretrained_model_type == 'LinearModel'):
        return Linear()
    elif(pretrained_model_type == 'ModdedLeNet5Net'):
        return ModdedLeNet5Net()
    elif(pretrained_model_type == 'badnet'):
        return BadNet()

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
        self.fcLayer = nn.Linear(1024*7*7, 2)

    def forward(self, image_in):
        #pdb.set_trace()
        x = self.featureEncoder(image_in)
        x = self.relu(x.view(x.shape[0], -1))
        x = self.fcLayer(x)
        return x


class AlexNet(nn.Module):
    """
    Modified AlexNet for CIFAR
    From: https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 2 * 2, 4096),
            nn.Linear(16384, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 2 * 2)
        x = x.view(x.size(0), 16384)
        x = self.classifier(x)
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck module in DenseNet Arch.
    See: https://arxiv.org/abs/1608.06993
    """
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        x = torch.cat([y, x], 1)
        return x


class Transition(nn.Module):
    """
    Transition module in DenseNet Arch.
    See: https://arxiv.org/abs/1608.06993
    """
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = F.avg_pool2d(x, 2)
        return x


class DenseNet(nn.Module):
    """
    From: https://github.com/icpm/pytorch-cifar10/blob/master/models/DenseNet.py
    """
    def __init__(self, block, num_block, growth_rate=12, reduction=0.5, num_classes=NUM_CLASSES):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = F.avg_pool2d(F.relu(self.bn(x)), 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


class ModdedLeNet5Net(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=3):
        super(ModdedLeNet5Net, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class BadNet(nn.Module):
    """
    Mnist network from BadNets paper
    Input - 1x28x28
    C1 - 1x28x28 (5x5 kernel) -> 16x24x24
    ReLU
    S2 - 16x24x24 (2x2 kernel, stride 2) Subsampling -> 16x12x12
    C3 - 16x12x12 (5x5 kernel) -> 32x8x8
    ReLU
    S4 - 32x8x8 (2x2 kernel, stride 2) Subsampling -> 32x4x4
    F6 - 512 -> 512
    tanh
    F7 - 512 -> 10 Softmax (Output)
    """

    def __init__(self):
        super(BadNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class Fullyconneted(nn.Module):

    def __init__(self):
        super(Fullyconneted, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(3*224*224 , 3000)  # 6*6 from image dimension

        self.dp1  = nn.Dropout(0.25)
        self.fc2 = nn.Linear(3000, 500)
        self.act1 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(250)
        self.dp2  = nn.Dropout(0.25)
        # self.relu = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU()
        # self.act = nn.Softmax()
        self.fc3 = nn.Linear(500, 20)
        self.act3 = nn.ReLU()
        # self.fc3 = nn.Linear(250, 50)
        self.fc4 = nn.Linear(20, 2)
        self.act4 = nn.ReLU()
    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = self.bn1(x)
        x =  x.view(x.size(0),-1)
        x = self.dp1(self.act1(self.fc1(x)))
        x = self.act2(self.fc2(x))
        x = self.dp2(x)
        x = self.act3(x)
        x = self.fc3(x)
        x = self.act4(x)
        x = self.fc4(x)
        return x


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(3*224*224 , 2)  # 6*6 from image dimension

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = self.bn1(x)
        x =  x.view(x.size(0),-1)
        x = self.fc1(x)

        return x
