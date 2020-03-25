import math
import torch
import torch.nn as nn
from torch.nn import init

#from layers import TemplateBank, SConv2d
from MoEIR.modules import TemplateBank, SConv2d

class SharedTemplateBank(nn.Module):
    def __init__(self, num_templates, out_feature_size, kernel_size):
        super(SharedTemplateBank, self).__init__()
        self.bank = TemplateBank(num_templates=num_templates, in_planes=out_feature_size, out_planes=out_feature_size, kernel_size=kernel_size) 

    def forward(self, x):
        pass
    

class SResidual_Block(nn.Module):
    def __init__(self, bank=None, res_scale = 0.1):
        super(SResidual_Block, self).__init__()
        
        self.conv1 = SConv2d(bank)
        self.conv2 = SConv2d(bank)
        self.relu = nn.ReLU(inplace=True)

        self.res_scale = res_scale
    

    def forward(self, x):
        residual = x
        
        x = self.conv2(self.relu(self.conv1(x)))
        x = x.mul(self.res_scale)
        
        x += residual
        return x



class SFEDSRNet(nn.Module):
    def __init__(self, bank, feature_size=256, out_feature_size=64, n_resblocks=7, n_templates=4, res_scale=1):
        super(SFEDSRNet, self).__init__()
        layers_per_bank = n_resblocks * 2 # Residual bock has 2 convolution layers
        kernel_size = 3 
        self.res_scale = res_scale
        self.bank = bank #shared TemplateBank
        self.res_block = self.make_layer(num_blocks=n_resblocks, bank=self.bank) 
        
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
       
        coefficient_inits = torch.zeros((layers_per_bank, n_templates, 1,1,1,1))
        nn.init.orthogonal_(coefficient_inits)
        for i in range(0, n_resblocks):
            self.res_block[i].conv1.coefficients.data = coefficient_inits[i*2]
            self.res_block[i].conv2.coefficients.data = coefficient_inits[(i*2)+1]
        
    def make_layer(self, num_blocks, bank):
        blocks = []
        for i in range(1, num_blocks+1):
            blocks.append(SResidual_Block(bank, res_scale=self.res_scale))
        return nn.Sequential(*blocks)  


    def forward(self, x):
        x = self.input(x) #head [featuresize -> out_featuresize]

        res = self.res_block(x) #body [out_featuresize -> out_featuresize]
        res += x
        # parameters in res_blocks are shared only        

        x = self.output(res)
        return x


    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"


        
