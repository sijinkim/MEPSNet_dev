import torch
import torch.nn as nn

from MoEIR.modules.feature_extractors import FeatureNet
from MoEIR.modules.experts import FVDSRNet
from MoEIR.modules.attentions import AttentionNet 
from MoEIR.modules.reconstructors import ChannelWiseAttentionNet


class MoE_with_Attention(nn.Module):
    def __init__(self, device, feature_size, expert_feature_size, n_experts, batch_size):
        super(MoE_with_Attention, self).__init__()
        
        self.batch = batch_size
        self.n_experts = n_experts

        self.feature_extractor = FeatureNet(feature_size=feature_size).to(device)
        self.experts = [FVDSRNet(feature_size=feature_size, out_feature_size=expert_feature_size).to(device) for _ in range(0, n_experts)]
        self.attention = AttentionNet(feature_size= expert_feature_size, num_experts=n_experts).to(device)
        self.reconstructor = ChannelWiseAttentionNet(in_channels=expert_feature_size, out_channels=3, num_experts=n_experts).to(device) 

    def forward(self, x):
        feature = self.feature_extractor(x)
        experts_output = [expert(feature) for expert in self.experts]
        concat_experts = torch.cat(tuple(experts_output), dim=1)
        cwa_output = self.attention(concat_experts)
        final_output = self.reconstructor(cwa_output)
        return final_output

    def take_modules(self):
        return [self.feature_extractor, self.experts, self.attention, self.reconstructor]
