import torch
import torch.nn as nn

class AttentionNet(nn.Module):
    def __init__(self, feature_size, num_experts):
        super(AttentionNet, self).__init__()
        concat_feature = feature_size * num_experts
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.squeeze = nn.Conv2d(in_channels=concat_feature,
                                 out_channels=concat_feature//2,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.excitation = nn.Conv2d(in_channels=concat_feature//2,
                                    out_channels=concat_feature,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        residual = x
        out = self.pool(x)
        out = self.relu(self.squeeze(out))
        out = self.sigmoid(self.excitation(out))
        out = residual * out
        return out

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"

class AttentionNet_in_RIR(nn.Module):
    def __init__(self, feature_size, num_experts):
        super(AttentionNet_in_RIR, self).__init__()
        concat_feature = feature_size * num_experts
        
        self.conv = nn.Conv2d(in_channels=concat_feature,
                              out_channels=concat_feature,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x

class PassNet(nn.Module):
    def __init__(self):
        super(PassNet, self).__init__()

    def forward(self, x):
        return x
    
    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"
