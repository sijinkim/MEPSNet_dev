#TODO: make experts trainig with gate modules
import torch
import torch.nn as nn

import pdb

class Train_with_gate(nn.Module):
    def __init__(self, module_sequence, batch_size, n_experts):
        super(Train_with_gate, self).__init__()
        self.modules = module_sequence
        self.batch = batch_size
        self.n_experts = n_experts
 
    def forward(self, x):
        expert_output = [expert(x) for expert in self.modules[0]]
        gate_output = self.modules[1](x)
        
        stacked_experts = torch.stack(expert_output, dim=1)
        weighted_feature = stacked_experts.mul(gate_output.view(self.batch, self.n_experts, 1, 1, 1)).sum(dim=1)
        final_output = weighted_feature

        return final_output
