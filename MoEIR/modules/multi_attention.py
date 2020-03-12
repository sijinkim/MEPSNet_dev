import torch
import torch.nn as nn

class GAP_GMP_AttentionNet(nn.Module):
    def __init__(self, feature_size, num_experts, gmp_k):
        super(GAP_GMP_AttentionNet, self).__init__()
        concat_feature = feature_size * num_experts

        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.squeeze = nn.Conv2d(in_channels=concat_feature,
                                 out_channels=concat_feature//2,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.excitation = nn.Conv2d(in_channels=concat_feature//2,
                                 out_channels=concat_feature,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        #self.GMP = nn.MaxPool2d((patch_size, patch_size), padding=0)

        self.detector = nn.Conv2d(in_channels=concat_feature,
                                  out_channels=gmp_k*num_experts,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)


        self.conv = nn.Conv2d(in_channels=concat_feature+ (gmp_k*num_experts),
                              out_channels=concat_feature,
                              kernel_size=1,
                              stride=1,
                              padding=0)       
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def GMP_ft(self, x):
        h, w = x.shape[-2], x.shape[-1]
        GMP = nn.MaxPool2d((h, w), padding=0)
        return GMP(x)
        
    def forward(self, x):
        residual = x #(feature_size*n_experts) x patch_size x patch_size
        gap_out = self.GAP(x)
        gap_out = self.relu(self.squeeze(gap_out))
        gap_out = self.excitation(gap_out) #(feature_size*n_experts) x 1 x 1

        gmp_out = self.relu(self.detector(x)) #(gmp_k*n_experts) x patch_size x patch_size
        gmp_out = self.GMP_ft(gmp_out)

        concat_feature = torch.cat(tuple([gap_out, gmp_out]), dim=1)
        final_out = self.sigmoid(self.conv(concat_feature)) #(feature_size*n_experts) x patch_size x patch_size
        final_out = residual * final_out
        return final_out
        
       
     
