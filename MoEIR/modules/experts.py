import math
import torch
import torch.nn as nn

class Conv_ReLU_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(Conv_ReLU_Block, self).__init__()
        self.body = nn.Sequential( nn.Conv2d(in_channels, 
                                             out_channels, 
                                             kernel_size, 
                                             stride, 
                                             padding, 
                                             bias=False),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.body(x)
        return output
 

class FVDSRNet(nn.Module):
    def __init__(self, feature_size=512, out_feature_size=64):
        super(FVDSRNet, self).__init__()
        self.conv_block = self.make_layer(Conv_ReLU_Block(in_channels = out_feature_size, 
                                                          out_channels = out_feature_size, 
                                                          kernel_size = 3, 
                                                          stride = 1, 
                                                          padding = 1), num_of_layer = 14)
        self.input = nn.Conv2d(in_channels = feature_size,
                               out_channels = out_feature_size,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.output = nn.Conv2d(in_channels = out_feature_size,
                               out_channels = out_feature_size,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)


    def forward(self, x):
        residual = self.input(x)
        output = self.conv_block(residual)
        output = self.output(output)
        output = torch.add(output, residual)
        return output


    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


class Residual_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 res_scale = 0.1):
        super(Conv_ReLU_Block, self).__init__()
        self.body = nn.Sequential( nn.Conv2d(in_channels, 
                                             out_channels, 
                                             kernel_size, 
                                             stride, 
                                             padding, 
                                             bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding,
                                             bias=False))
        self.res_scale = res_scale
    

    def forward(self, x):
        output = self.body(x).mul(self.res_scale)
        output += x
        return output



class FEDSRNet(nn.Module):
    def __init__(self, feature_size=512, out_feature_size=64):
        super(FEDSRNet, self).__init__()
        self.res_block = self.make_layer(Residual_Block(in_channels=out_feature_size, 
                                                        out_channels=out_feature_size, 
                                                        kernel_size=3, 
                                                        stride=1, 
                                                        padding=1,
                                                        res_scale=0.1), num_of_layer = 10)
        self.input = nn.Conv2d(in_channels=feature_size,
                               out_channels=out_feature_size,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.output = nn.Conv2d(in_channels=out_feature_size,
                               out_channels=out_feature_size,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)


    def forward(self, x):
        residual = self.input(x)
        output = self.res_block(residual)
        output = self.output(output)
        output = torch.add(output, residual)
        return output


    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


