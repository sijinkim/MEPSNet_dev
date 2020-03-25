import torch
import torch.nn as nn

from MoEIR.modules import FeatureNet
from MoEIR.modules import FVDSRNet, FEDSRNet
from MoEIR.modules import AttentionNet 
from MoEIR.modules import ReconstructNet
from MoEIR.modules import GAP_GMP_AttentionNet

class MoE_with_Attention(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_Attention, self).__init__()
        
        self.n_experts = n_experts
        self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device)
        
        self.reconstructor = ReconstructNet(in_channels=args.ex_featuresize, out_channels=3, num_experts=n_experts).to(device) 
        
        if args.multi_attention:
            self.attention = GAP_GMP_AttentionNet(feature_size=args.ex_featuresize, num_experts=n_experts, gmp_k=args.gmp_k).to(device)
        elif not args.multi_attention:
            self.attention = AttentionNet(feature_size= args.ex_featuresize, num_experts=n_experts).to(device)
        else: 
            print(f"ValueError: multi_attention {args.multi_attention}") 
            raise ValueError        
   

        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
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
        sequence = {'feature_extractor': self.feature_extractor, 
                    'experts': self.experts, 
                    'attention': self.attention, 
                    'reconstructor': self.reconstructor}
        return sequence
