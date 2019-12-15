import torch.nn as nn


class ReconstructNet(nn.Module):
    def __init__(self,
                 in_channels = 64,
                 out_channels = 3):
        super(ReconstructNet, self).__init__()
        self.recon = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = False)

    def forward(self, x):
        output = self.recon(x)
        return output

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


class ChannelWiseAttentionNet(nn.Module):
    def __init__(self,
                 in_channels = 64,
                 out_channels = 3,
                 num_experts = 2):
        super(ChannelWiseAttentionNet, self).__init__()
        feature_size = in_channels * num_experts
        self.recon = nn.Sequential(nn.Conv2d(in_channels=feature_size,
                                             out_channels=feature_size//2,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias = False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=feature_size//2,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False))

    def forward(self, x):
        output = self.recon(x)
        return output

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"
