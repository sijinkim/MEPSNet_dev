import math
import torch
import torch.nn as nn

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels=64, 
                                            out_channels=64, 
                                            kernel_size=3, 
                                            stride=3,
                                            padding=1,
                                            bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False))
        self.res_scale = 0.1 

    def forward(self, x):
        output = self.body(x).mul(self.res_scale)
        output += x
        return output


class EDSRNet(nn.Module):
    def __init__(self, num_blocks=10, num_experts=1):
        super(EDSRNet, self).__init__()
        self.input = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.output = nn.Conv2d(in_channels=64,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        
        self.res_block = self.make_layer(Residual_Block(), num_of_layer=num_blocks * num_experts)


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
        x = self.input(x)
        res = self.res_block(x)
        res += x
        output = self.output(res)
        return output

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"
