import math
import torch
import torch.nn as nn
from torch.nn import init

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
                                             bias=True),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.body(x)
        return output
 

class FVDSRNet(nn.Module):
    def __init__(self, feature_size=512, out_feature_size=64, kernel_size=3, num_of_layer=14):
        super(FVDSRNet, self).__init__()
        self.kernel = kernel_size
        padding = 0
        if self.kernel == 1:
            padding = 0
        elif self.kernel == 3:
            padding = 1
        elif self.kernel == 5:
            padding = 2
        elif self.kernel == 7:
            padding = 3
        else:
            raise ValueError

        self.layers = num_of_layer
        self.conv_block = self.make_layer(Conv_ReLU_Block(in_channels = out_feature_size, 
                                                          out_channels = out_feature_size, 
                                                          kernel_size = self.kernel, 
                                                          stride = 1, 
                                                          padding = padding), num_of_layer=self.layers)
        self.input = nn.Conv2d(in_channels = feature_size,
                               out_channels = out_feature_size,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = True)
        self.output = nn.Conv2d(in_channels = out_feature_size,
                               out_channels = out_feature_size,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = True)
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
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 device=None):
        super(Residual_Block, self).__init__()
#        self.body = nn.Sequential( nn.Conv2d(in_channels, 
#                                             out_channels, 
#                                             kernel_size, 
#                                             stride, 
#                                             padding, 
#                                             bias=True),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv2d(in_channels,
#                                             out_channels,
#                                             kernel_size,
#                                             stride,
#                                             padding,
#                                             bias=True))
#    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True).to(device)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)


    def forward(self, x):
        res = x
        x = self.conv2(self.relu(self.conv1(x)))
        x += res
        return x

class RIR(nn.Module):
    def __init__(self, n_resblocks_in_one_Block=12, feature=64, kernel=3, device=None):
        super(RIR, self).__init__()
        self.n_resblocks = n_resblocks_in_one_Block

        blocks = []
        for _ in range(0, self.n_resblocks):
            blocks.append(Residual_Block(in_channels=feature,
                                         out_channels=feature,
                                         kernel_size=kernel,
                                         stride=1,
                                         padding=kernel//2))

        self.res_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        res = x
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x += res#RIR
        return x


class FEDSRNet(nn.Module):
    def __init__(self, feature_size=512, out_feature_size=64, kernel_size=3, n_residual_blocks=12, device=None):
        super(FEDSRNet, self).__init__()
        self.out_feature = out_feature_size
        self.n_resblocks = n_residual_blocks
        self.kernel = kernel_size
         
#        self.res_block = self.make_layer(Residual_Block(in_channels=out_feature_size, 
#                                                        out_channels=out_feature_size, 
#                                                        kernel_size=self.kernel, 
#                                                        stride=1, 
#                                                        padding=(kernel_size//2),
#                                                        res_scale=0.1), n_resblocks_in_one_Block = n_residual_block)
        self.res_block = self.make_layer_rir(device=device)


        self.input = nn.Conv2d(in_channels=feature_size,
                               out_channels=out_feature_size,
                               kernel_size=3,
                               stride=1,
                               padding=(kernel_size//2),
                               bias=True)
        self.output = nn.Conv2d(in_channels=out_feature_size,
                               out_channels=out_feature_size,
                               kernel_size=3,
                               stride=1,
                               padding=(kernel_size//2),
                               bias=True)
    
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)

    def make_layer_rir(self, device):
        n_Blocks = 3 #Fix 3 Residual learning blocks
        layers = []
        for i in range(0,n_Blocks):
            print(f'Make RIR fedsr block {i} - # of residual blocks in one Block: {self.n_resblocks}')
            layers.append(RIR(feature=self.out_feature, 
                              n_resblocks_in_one_Block=self.n_resblocks,
                              kernel=self.kernel,
                              device=device))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.input(x) #head
        res = self.res_block(x) #body
        res += x

        x = self.output(res)
        return x


    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"



