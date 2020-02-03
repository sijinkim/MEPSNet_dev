import torch
import torch.nn as nn

from MoEIR.baseline import VDSRNet, EDSRNet

class BaselineNet(nn.Module):
    def __init__(self, device, n_experts, model_type):
        super(BaselineNet, self).__init__()
        
        self.model_type = model_type
        if self.model_type == 'vdsr':
            self.net = VDSRNet(num_experts=n_experts).to(device)
        elif self.model_type == 'edsr':
            self.net = EDSRNet(num_experts=n_experts).to(device)
        else:
            raise ValueError
    

    def forward(self, x):
        output = self.net(x)
        return output
