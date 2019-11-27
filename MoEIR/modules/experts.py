import torch.nn as nn


class FVDSRNet(nn.Module):
    def __init__(self, feature_size=512):
        super(FVDSRNet, self).__init__()

        # TODO: Implement the layers, Note that defualt feature_size is 512.

    def forward(self, x):
        output = x
        return output

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"
