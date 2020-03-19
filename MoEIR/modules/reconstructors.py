import torch.nn as nn

from .common import MeanShift

class ReconstructNet(nn.Module):
    def __init__(self,
                 in_channels = 64,
                 out_channels = 3):
        super(ReconstructNet, self).__init__()
        self.add_mean = MeanShift(rgb_range=255, sign=1)
        self.recon = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = True)

    def forward(self, x):
        x = self.recon(x)
        output = self.add_mean(x)
        return output

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


class ReconstructNet_with_CWA(nn.Module):
    def __init__(self,
                 in_channels = 64,
                 out_channels = 3,
                 num_experts = 2):
        super(ReconstructNet_with_CWA, self).__init__()
        self.add_mean = MeanShift(rgb_range=255, sign=1)
        
        feature_size = in_channels * num_experts
        self.recon = nn.Sequential(nn.Conv2d(in_channels=feature_size,
                                             out_channels=feature_size//2,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias = True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=feature_size//2,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=True))

    def forward(self, x):
        x = self.recon(x)
        output = self.add_mean(x)
        return output

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"
