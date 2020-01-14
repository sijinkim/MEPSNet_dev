import torch
import torch.nn as nn

from MoEIR.modules import FeatureNet
from MoEIR.modules import FVDSRNet, FEDSRNet
from MoEIR.modules import AttentionNet 
from MoEIR.modules import ReconstructNet_with_CWA


class MoE_with_Attention(nn.Module):
    def __init__(self, device, feature_size, expert_feature_size, n_experts, experts_type, batch_size):
        super(MoE_with_Attention, self).__init__()
        
        self.batch = batch_size
        self.n_experts = n_experts

        self.feature_extractor = FeatureNet(feature_size=feature_size).to(device)
        
        self.attention = AttentionNet(feature_size= expert_feature_size, num_experts=n_experts).to(device)
        self.reconstructor = ReconstructNet_with_CWA(in_channels=expert_feature_size, out_channels=3, num_experts=n_experts).to(device) 

        ex_type = experts_type
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=feature_size, out_feature_size=expert_feature_size).to(device) for _ in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=feature_size, out_feature_size=expert_feature_size).to(device) for _ in range(0, n_experts)]
        else:
            raise ValueError
            

    def forward(self, x):
        feature = self.feature_extractor(x)
        experts_output = [expert(feature) for expert in self.experts]
        concat_experts = torch.cat(tuple(experts_output), dim=1)
        cwa_output = self.attention(concat_experts)
        final_output = self.reconstructor(cwa_output)
        return final_output

    def forward_valid_phase(self, x):
        feature = self.feature_extractor(x)
        experts_output = [expert(feature) for expert in self.experts]
        concat_experts = torch.cat(tuple(experts_output), dim=1)
        cwa_output = self.attention(concat_experts)
        final_output = self.reconstructor(cwa_output)
        return final_output

    def take_modules(self):
        return [self.feature_extractor, self.experts, self.attention, self.reconstructor]
