import torch
import torch.nn as nn
from torchvision.models import resnet34

from .common import MeanShift

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet34(pretrained=True)

        # Drop the last fc layers. B x 512 x H x W
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-2])

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"

class FeatureNet(nn.Module):
    def __init__(self, feature_size=64): 
        super(FeatureNet, self).__init__()

        self.sub_mean = MeanShift(rgb_range=255)

        self.input = nn.Conv2d(in_channels=3,
                               out_channels=feature_size//2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.conv1 = nn.Conv2d(in_channels=feature_size//2,
                               out_channels=feature_size,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=feature_size,
                               out_channels=feature_size,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sub_mean(x)

        output = self.relu(self.input(x))
        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))

        return output
       

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


class LiteFeatureNet(nn.Module):
    def __init__(self, feature_size=64): 
        super(LiteFeatureNet, self).__init__()
        kernel = 3
        self.sub_mean = MeanShift(rgb_range=255)

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=feature_size,
                               kernel_size=kernel,
                               stride=1,
                               padding=(kernel//2),
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sub_mean(x)

        output = self.relu(self.conv1(x))

        return output
       

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


