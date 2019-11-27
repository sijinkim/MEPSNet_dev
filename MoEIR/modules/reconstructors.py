import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


class ChannelWiseAttentionNet(nn.Module):
    def __init__(self):
        super(ChannelWiseAttentionNet, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"
