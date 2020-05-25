import torch
import torch.nn as nn

from MoEIR.modules import FeatureNet, LiteFeatureNet
from MoEIR.modules import FVDSRNet, FEDSRNet
from MoEIR.modules import SharedTemplateBank, SFEDSRNet, SFEDSRNet_noLSC
from MoEIR.modules import AttentionNet, PassNet, AttentionNet_in_RIR
from MoEIR.modules import ReconstructNet, LiteReconstructNet
from MoEIR.modules import GAP_GMP_AttentionNet


class MoE_with_Template(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_Template, self).__init__()
        
        self.n_experts = n_experts
            
        # Set feature extraction network        
        if args.lite_feature:
            self.feature_extractor = LiteFeatureNet(feature_size=args.featuresize).to(device)
        elif not args.lite_feature:
            self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device)
        else:
            print(f"ValueError: Check the argument lite_feature {args.lite_feature}")
            raise ValueError
 
        # Set reconstruction network
        if args.lite_reconst:
            self.reconstructor = LiteReconstructNet(in_channels=args.ex_featuresize, 
                                                    out_channels=3, 
                                                    num_experts=n_experts).to(device)
        elif not args.lite_reconst:
            self.reconstructor = ReconstructNet(in_channels=args.ex_featuresize, 
                                                out_channels=3, 
                                                num_experts=n_experts).to(device)
        else:
            print(f"ValueError: Check the argument reconst network {args.lite_reconst}")
            raise ValueError
        
        # Set attention network
        if args.multi_attention:
            self.attention = GAP_GMP_AttentionNet(feature_size=args.ex_featuresize,
                                                  num_experts=n_experts, 
                                                  gmp_k=args.gmp_k).to(device)
        elif not args.multi_attention:
            self.attention = AttentionNet(feature_size=args.ex_featuresize, 
                                          num_experts=n_experts).to(device)
        else: 
            print(f"ValueError: Check the argument multi_attention {args.multi_attention}") 
            raise ValueError        
   
        # Set type of experts network
        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'sfedsr':
            self.template_bank = SharedTemplateBank(num_bank=args.n_bank, num_templates=args.n_template, first_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[0], device=device).to(device)

            self.experts = [SFEDSRNet(bank=self.template_bank.bank, feature_size=args.featuresize, out_feature_size=args.ex_featuresize, n_resblocks=args.n_resblock, n_templates=args.n_template, device=device, rir=args.rir).to(device) for _ in range(0, n_experts)]
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


class MoE_with_Template_without_CWA(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_Template, self).__init__()
        
        self.n_experts = n_experts
            
        # Set feature extraction network        
        if args.lite_feature:
            self.feature_extractor = LiteFeatureNet(feature_size=args.featuresize).to(device)
        elif not args.lite_feature:
            self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device)
        else:
            print(f"ValueError: Check the argument lite_feature {args.lite_feature}")
            raise ValueError
 
        # Set reconstruction network
        if args.lite_reconst:
            self.reconstructor = LiteReconstructNet(in_channels=args.ex_featuresize, 
                                                    out_channels=3, 
                                                    num_experts=n_experts).to(device)
        elif not args.lite_reconst:
            self.reconstructor = ReconstructNet(in_channels=args.ex_featuresize, 
                                                out_channels=3, 
                                                num_experts=n_experts).to(device)
        else:
            print(f"ValueError: Check the argument reconst network {args.lite_reconst}")
            raise ValueError
        
        # Set attention network
        self.attention = PassNet() 
   
        # Set type of experts network
        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'sfedsr':
            self.template_bank = SharedTemplateBank(num_templates=args.n_template, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[0]).to(device)
            self.experts = [SFEDSRNet(bank=self.template_bank.bank, feature_size=args.featuresize, out_feature_size=args.ex_featuresize, n_resblocks=args.n_resblock, n_templates=args.n_template, res_scale=args.res_scale).to(device) for i in range(0, n_experts)] 
        else:
            raise ValueError

    def forward(self, x):
        feature = self.feature_extractor(x)
        experts_output = [expert(feature) for expert in self.experts]
        concat_experts = torch.cat(tuple(experts_output), dim=1)
        final_output = self.reconstructor(concat_experts)
        return final_output

    def forward_valid_phase(self, x):
        feature = self.feature_extractor(x)
        experts_output = [expert(feature) for expert in self.experts]
        concat_experts = torch.cat(tuple(experts_output), dim=1)
        final_output = self.reconstructor(concat_experts)
        return final_output

    def take_modules(self):
        sequence = {'feature_extractor': self.feature_extractor, 
                    'experts': self.experts, 
                    'attention': self.attention, 
                    'reconstructor': self.reconstructor}
        return sequence


class MoE_with_Template_CWA_in_RIR(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_Template_CWA_in_RIR, self).__init__()
        
        self.n_experts = n_experts
            
        # Set feature extraction network        
        if args.lite_feature:
            self.feature_extractor = LiteFeatureNet(feature_size=args.featuresize).to(device)
        elif not args.lite_feature:
            self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device)
        else:
            print(f"ValueError: Check the argument lite_feature {args.lite_feature}")
            raise ValueError
 
        # Set reconstruction network
        if args.lite_reconst:
            self.reconstructor = LiteReconstructNet(in_channels=args.ex_featuresize, 
                                                    out_channels=3, 
                                                    num_experts=n_experts).to(device)
        elif not args.lite_reconst:
            self.reconstructor = ReconstructNet(in_channels=args.ex_featuresize, 
                                                out_channels=3, 
                                                num_experts=n_experts).to(device)
        else:
            print(f"ValueError: Check the argument reconst network {args.lite_reconst}")
            raise ValueError
        
        if args.conv_fusion:
            self.attention = AttentionNet_in_RIR(feature_size=args.ex_featuresize, num_experts=n_experts).to(device)

        elif args.cwa_fusion:
            self.attention = AttentionNet(feature_size=args.ex_featuresize, num_experts=n_experts).to(device)

        else: 
            print(f"ValueError: Check the argument about Feature fusion") 
            raise ValueError        
   
        # Set type of experts network
        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i], n_residual_blocks=args.n_sres, device=device).to(device) for i in range(0, n_experts)]
        elif ex_type == 'sfedsr':
            self.template_bank = SharedTemplateBank(num_bank=args.n_bank, num_templates=args.n_template, first_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[0], device=device).to(device)

            self.experts = [SFEDSRNet(bank=self.template_bank.bank, feature_size=args.featuresize, out_feature_size=args.ex_featuresize, n_resblocks=args.n_resblock, n_templates=args.n_template, device=device, RIRintoBlock=args.RIRintoBlock, rir_cwa=args.rir_attention, sresblocks_in_rir=args.n_sres, dilate=args.is_dilate).to(device) for _ in range(0, n_experts)]
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


class MoE_with_LSC(nn.Module):
    def __init__(self, device, n_experts, args):
        super(MoE_with_LSC, self).__init__()
        #MoE_with_LSC - experts: SFEDSRNet_noLSC
        print("MoE_with_LSC expert module - There are Long skip connection between 3 channel images.") 
        self.n_experts = n_experts
            
        # Set feature extraction network        
        if args.lite_feature:
            self.feature_extractor = LiteFeatureNet(feature_size=args.featuresize).to(device)
        elif not args.lite_feature:
            self.feature_extractor = FeatureNet(feature_size=args.featuresize).to(device)
        else:
            print(f"ValueError: Check the argument lite_feature {args.lite_feature}")
            raise ValueError
 
        # Set reconstruction network
        if args.lite_reconst:
            self.reconstructor = LiteReconstructNet(in_channels=args.ex_featuresize, 
                                                    out_channels=3, 
                                                    num_experts=n_experts).to(device)
        elif not args.lite_reconst:
            self.reconstructor = ReconstructNet(in_channels=args.ex_featuresize, 
                                                out_channels=3, 
                                                num_experts=n_experts).to(device)
        else:
            print(f"ValueError: Check the argument reconst network {args.lite_reconst}")
            raise ValueError
        
        if args.conv_fusion:
            self.attention = AttentionNet_in_RIR(feature_size=args.ex_featuresize, num_experts=n_experts).to(device)

        elif args.cwa_fusion:
            self.attention = AttentionNet(feature_size=args.ex_featuresize, num_experts=n_experts).to(device)

        else: 
            print(f"ValueError: Check the argument about Feature fusion") 
            raise ValueError        
   
        # Set type of experts network
        ex_type = args.experts[0]
        if ex_type == 'fvdsr':
            self.experts = [FVDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i]).to(device) for i in range(0, n_experts)]
        elif ex_type == 'fedsr':
            self.experts = [FEDSRNet(feature_size=args.featuresize, out_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[i], n_residual_blocks=args.n_sres, device=device).to(device) for i in range(0, n_experts)]
        elif ex_type == 'sfedsr':
            self.template_bank = SharedTemplateBank(num_bank=args.n_bank, num_templates=args.n_template, first_feature_size=args.ex_featuresize, kernel_size=args.kernelsize[0], device=device).to(device)

            self.experts = [SFEDSRNet_noLSC(bank=self.template_bank.bank, feature_size=args.featuresize, out_feature_size=args.ex_featuresize, n_resblocks=args.n_resblock, n_templates=args.n_template, device=device, RIRintoBlock=args.RIRintoBlock, rir_cwa=args.rir_attention, sresblocks_in_rir=args.n_sres, dilate=args.is_dilate).to(device) for _ in range(0, n_experts)]
        else:
            raise ValueError


    def forward(self, x):
        residual = x
        feature = self.feature_extractor(x)
        experts_output = [expert(feature) for expert in self.experts]
        concat_experts = torch.cat(tuple(experts_output), dim=1)
        cwa_output = self.attention(concat_experts)
        output = self.reconstructor(cwa_output)
        final_output = output + residual
    
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

