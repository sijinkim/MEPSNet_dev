import torch
import torch.nn as nn

from MoEIR.modules  import FeatureNet
from MoEIR.modules import FVDSRNet, FEDSRNet
from MoEIR.modules import ReconstructNet

class MoE_with_Gate(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_Gate, self).__init__()

        self.batch = args.batchsize
        self.n_experts = n_experts
        
        self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device) 
        self.reconstructor = ReconstructNet(in_channels=args.ex_featuresize, out_channels=3).to(device)

        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        else:
            raise ValueError 

        gate_key = args.gate
        if gate_key == 'gmp':
            from MoEIR.modules import GMP_GateNet
            self.gate = GMP_GateNet(in_feature_size=args.featuresize, out_feature_size=args.ex_featuresize, num_experts=n_experts).to(device)
        elif gate_key == 'gap':
            from MoEIR.module import GAP_GateNet
            self.gate = GAP_GateNet(in_feature_size=args.featuresize, out_feature_size=args.ex_featuresize, num_experts=n_experts).to(device)
        else:
            raise ValueError

         
    def forward(self, x):
        feature = self.feature_extractor(x)
        expert_output = [expert(feature) for expert in self.experts]
        gate_output = self.gate(feature)
        
        stacked_experts = torch.stack(expert_output, dim=1)
        weighted_feature = stacked_experts.mul(gate_output.view(self.batch, self.n_experts, 1, 1, 1)).sum(dim=1)
        final_output = self.reconstructor(weighted_feature)
        return final_output

    def forward_valid_phase(self,x):
        feature = self.feature_extractor(x)
        expert_output = [expert(feature) for expert in self.experts]
        gate_output = self.gate(feature)
        
        stacked_experts = torch.stack(expert_output, dim=1)
        weighted_feature = stacked_experts.mul(gate_output.view(1, self.n_experts, 1, 1, 1)).sum(dim=1)
        final_output = self.reconstructor(weighted_feature)
        return final_output


    def take_modules(self):
        sequence =  {'feature_extractor': self.feature_extractor, 
                     'experts': self.experts, 
                     'gate': self.gate, 
                     'reconstructor': self.reconstructor}
        return sequence
