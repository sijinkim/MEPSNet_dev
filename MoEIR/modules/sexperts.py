import math
import torch
import torch.nn as nn
from torch.nn import init

from MoEIR.modules import TemplateBank, SConv2d

class SharedTemplateBank(nn.Module):
    def __init__(self, num_templates, first_feature_size, kernel_size, num_bank=3, device=None):
        super(SharedTemplateBank, self).__init__()
        
        channels = first_feature_size
        self.bank = []
        for _ in range(0, num_bank):
            self.bank.append(TemplateBank(num_templates=num_templates, in_planes=channels, out_planes=channels, kernel_size=kernel_size).to(device))
            channels += channels 
        
    def forward(self, x):
        pass
    

class SResidual_Block(nn.Module):
    def __init__(self, bank=None, device= None):
        super(SResidual_Block, self).__init__()
        
        self.conv1 = SConv2d(bank).to(device)
        self.conv2 = SConv2d(bank).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)

    def forward(self, x):
        res = x
        x = self.conv2(self.relu(self.conv1(x)))
        #x = x.mul(self.res_scale)
        x += res
        return x

class RIR(nn.Module):
    def __init__(self, n_block=3, bank=None, device=None):
        super(RIR, self).__init__()
        self.bank = bank
        
        blocks = []
        for _ in range(0, n_block):
            blocks.append(SResidual_Block(bank, device))
        
        self.Sres_blocks = nn.ModuleList(blocks)
        
    def forward(self, x):
        res = x

        for res_block in self.Sres_blocks:
            x = res_block(x)

        x += res #RIR
        return x

        

class PSRDB(nn.Module):
    def __init__(self, n_block=3, ex_feature_size=64, bank=None, device=None):
        super(PSRDB, self).__init__()
        self.bank_list = bank
        self.relu = nn.ReLU(inplace=True).to(device)
        self.conv1_list = []
        self.conv2_list = []
        
        concat_size = ex_feature_size
        for i in range(0, n_block): #number of resblock in one PSRDB
            self.conv1_list.append(SConv2d(self.bank_list[i]).to(device))
            self.conv2_list.append(SConv2d(self.bank_list[i]).to(device))
            
            concat_size += concat_size
        
        self.channel_conv = nn.Conv2d(in_channels=concat_size,
                                      out_channels=ex_feature_size,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True)
    
    def forward(self, x):
        inputs = []
        inputs.append(x)
        # Fix n_block 3
        block0 = self.conv2_list[0](self.relu(self.conv1_list[0](inputs[0])))
        inputs.append(torch.cat((block0, inputs[0]), dim=1)) #inputs[1]

        block1 = self.conv2_list[1](self.relu(self.conv1_list[1](inputs[1])))
        inputs.append(torch.cat((block1, block0, inputs[0]), dim=1)) #inputs[2]

        block2 = self.conv2_list[2](self.relu(self.conv1_list[2](inputs[2])))
        inputs.append(torch.cat((block2, block1, block0, inputs[0]), dim=1)) #inputs[3]

        fusion_feature = inputs[-1]
        result = self.channel_conv(fusion_feature)
        final_output = result + inputs[0]
        return final_output
        
 
class SFEDSRNet(nn.Module):
    def __init__(self, bank, feature_size=256, out_feature_size=64, n_resblocks=9, n_templates=4, device=None, rir=False):
        super(SFEDSRNet, self).__init__()
        kernel_size = 3
        if len(bank) == 1:
            #TODO: Single bank parameter shared FEDSRNet
            layers_per_bank = n_resblocks * 2
            self.bank = bank
            if not rir:
                self.res_block = self.make_layer_Sres(num_blocks=n_resblocks, bank=self.bank[0], device=device)
                
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        init.kaiming_normal_(m.weight)

                coefficient_inits = torch.zeros((layers_per_bank, n_templates,1,1,1,1))
                nn.init.orthogonal_(coefficient_inits)
                for i in range(0, n_resblocks):
                    self.res_block[i].conv1.coefficients.data = coefficient_inits[i*2].to(device)
                    self.res_block[i].conv2.coefficients.data = coefficient_inits[(i*2)+1].to(device)  
 
            elif rir:
                self.res_block = self.make_layer_Srir(num_blocks=int(n_resblocks/3), bank=self.bank[0], device=device)
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        init.kaiming_normal_(m.weight)
                
                coefficient_inits = torch.zeros((layers_per_bank, n_templates, 1,1,1,1))
                nn.init.orthogonal_(coefficient_inits)

                count = 0
                for i in range(0, len(self.res_block)):
                    for j in range(0, len(self.res_block[i].Sres_blocks)):
                        self.res_block[i].Sres_blocks[j].conv1.coefficients.data = coefficient_inits[count].to(device)
                        count += 1
                        self.res_block[i].Sres_blocks[j].conv2.coefficients.data = coefficient_inits[count].to(device)
                        count += 1

            else:
                print(f"ValueError: Check the resblock type. without RIR or with RIR?")
                raise ValueError

                   
        elif len(bank) == 3:
            #TODO: multiple banks parameter shared FEDSRNet - Fix 3 psrdbs in one SFEDSRNet.
            num_rdb = 3
            layers_per_experts = n_resblocks * 2
            layers_per_rdb = layers_per_experts / num_rdb # convolutional layers in one PSRDB(default: 6)
        
            layers_per_bank = 2 * num_rdb  # 2 * num_rdb => 6
            resblocks_per_rdb = layers_per_rdb / 2 # SResidual_blocks in one PSRDB(default: 3)

            self.bank = bank #shared TemplateBanks
            self.res_block = self.make_layer(n_rdb=num_rdb, n_block=resblocks_per_rdb, feature_size=out_feature_size, banks=self.bank, device=device) 
        
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight)
            print('PSRDB Blocks: ', len(self.res_block), '\nSRes Blocks in one PSRDB: ', resblocks_per_rdb) 
            coefficient_inits = torch.zeros((layers_per_bank, n_templates, 1, 1, 1 ,1))
            nn.init.orthogonal_(coefficient_inits)
            for i in range(0, len(self.res_block)):
                for j in range(0, int(resblocks_per_rdb)):
                    self.res_block[i].conv1_list[j].coefficients.data = coefficient_inits[i*2].to(device)
                    self.res_block[i].conv2_list[j].coefficients.data = coefficient_inits[(i*2)+1].to(device)

 
        else:
            print("ValueError: Give argument n_bank:1 or 3. 1 - single bank model. 3 - three banks with Parameter Shared Residual Dense Block(PSRDB)")
            raise ValueError        
        

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
  
    def make_layer_Sres(self, num_blocks, bank, device):
        blocks = []
        for i in range(0, num_blocks):
            blocks.append(SResidual_Block(bank, device)) 
        return nn.Sequential(*blocks)

    
    def make_layer_Srir(self, num_blocks, bank, device):
        blocks = []
        for i in range(0, num_blocks):
            blocks.append(RIR(3, bank, device)) # 3 SResidual blocks in one RIR
        return nn.Sequential(*blocks)


    def make_layer(self, n_rdb, n_block, feature_size, banks, device):
        blocks = []
        for _ in range(0, n_rdb):
            blocks.append(PSRDB(int(n_block), feature_size, banks, device))
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


        
