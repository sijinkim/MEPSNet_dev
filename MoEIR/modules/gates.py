import torch
import torch.nn as nn

#GateNet with Global Max Pooling
class GMP_GateNet(nn.Module):
    def __init__(self,
                 in_feature_size = 512,
                 out_feature_size = 64,
                 num_experts = 2):
        super(GMP_GateNet, self).__init__()
        self.input = nn.Conv2d(in_channels = in_feature_size,                                                                                             out_channels = out_feature_size, 
                               kernel_size = 3,
                               stride = 2,
                               padding = 1) 
        self.output = nn.Conv2d(in_channels = out_feature_size // 2,
                                out_channels = out_feature_size // 4,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1)
        self.conv1 = nn.Conv2d(in_channels = out_feature_size,
                               out_channels = out_feature_size // 2,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        self.conv2 = nn.Conv2d(in_channels = out_feature_size // 2,
                               out_channels = out_feature_size // 2,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        self.relu = nn.ReLU(inplace = True)
        self.pool = nn.AdaptiveMaxPool2d((5,5))
        self.fc1 = nn.Linear((out_feature_size // 4) * 5 * 5, num_experts)
        self.final_size = out_feature_size // 4

    def forward(self, x):
        output = self.relu(self.input(x))
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.pool(self.output(output))
        output = output.view(-1, self.final_size * 5 * 5)
        output = self.fc1(output)
        return torch.sigmoid(output)

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"

#GateNet with Global Average Pooling
class GAP_GateNet(nn.Module):
    def __init__(self,
                 in_feature_size = 512,
                 out_feature_size = 64,
                 num_experts = 2):
        super(GAP_GateNet, self).__init__()
        self.input = nn.Conv2d(in_channels = in_feature_size,
                               out_channels = out_feature_size, 
                               kernel_size = 3,
                               stride = 2,
                               padding = 1) 
        self.output = nn.Conv2d(in_channels = out_feature_size // 2,
                                out_channels = out_feature_size // 4,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1)
        self.conv1 = nn.Conv2d(in_channels = out_feature_size,
                               out_channels = out_feature_size // 2,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        self.conv2 = nn.Conv2d(in_channels = out_feature_size // 2,
                               out_channels = out_feature_size // 2,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        self.relu = nn.ReLU(inplace = True)
        self.pool = nn.AdaptiveAvgPool2d((5,5))
        self.fc1 = nn.Linear((out_feature_size // 4) * 5 * 5, num_experts)
        self.final_size = out_feature_size // 4

    def forward(self, x):
        output = self.relu(self.input(x))
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.pool(self.output(output))
        output = output.view(-1, self.final_size * 5 * 5)
        output = self.fc1(output)
        return torch.sigmoid(output)

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


