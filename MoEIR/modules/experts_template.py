import torch
import torch.nn as nn

from MoEIR.modules import FeatureNet
from MoEIR.modules import FVDSRNet, FEDSRNet
from MoEIR.modules import SharedTemplateBank, SFEDSRNet
from MoEIR.modules import AttentionNet 
from MoEIR.modules import ReconstructNet_with_CWA
from MoEIR.modules import GAP_GMP_AttentionNet


class MoE_with_Template(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_Template, self).__init__()
        
        self.n_experts = n_experts
        self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device)
        self.reconstructor = ReconstructNet_with_CWA(in_channels=args.ex_featuresize, out_channels=3, num_experts=n_experts).to(device) 
        
        if args.multi_attention:
            self.attention = GAP_GMP_AttentionNet(feature_size=args.ex_featuresize, num_experts=n_experts, gmp_k=args.gmp_k).to(device)
        elif not args.multi_attention:
            self.attention = AttentionNet(feature_size= args.ex_featuresize, num_experts=n_experts).to(device)
        else: 
            print(f"ValueError: Check the argument multi_attention {args.multi_attention}") 
            raise ValueError        
   
        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'sfedsr':
            # No change in kernel size
            self.template_bank = SharedTemplateBank(num_templates=args.n_template, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[0]).to(device)
            self.experts = [SFEDSRNet(bank=self.template_bank.bank, feature_size=args.featuresize, out_feature_size=args.ex_featuresize, n_resblocks=args.n_resblock, n_templates=args.n_template).to(device) for i in range(0, n_experts)] 
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
