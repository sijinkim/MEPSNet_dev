def prepare_modules(module_map, device, feature_size, expert_feature_size, num_experts):
    module_seq = []
    for module_key in module_map.keys():
        if module_key == 'feature_extractor':
            module = get_feature_extractor_module(module_map[module_key], feature_size)
            module = module.to(device)

        elif module_key == 'experts':
            # Because we need the multiple experts,
            # module, here, should be a list of the experts.
            module = []
            for expert_key in module_map[module_key]:
                module.append(get_expert_module(expert_key, feature_size, expert_feature_size).to(device))

        elif module_key == 'gate':
            module = get_gate_module(module_map[module_key], feature_size, expert_feature_size, num_experts).to(device)

        elif module_key == 'reconstructor':
            module = get_recon_module(module_map[module_key], expert_feature_size, num_experts).to(device)

        elif module_key == 'attention':
            module = get_attention_module(module_map[module_key]).to(device)
        
        else:
            raise ValueError
        
        module_seq.append(module)

    return module_seq


def get_feature_extractor_module(extractor_key, f_size):
    if extractor_key == 'resnet':
        from MoEIR.modules.feature_extractors import ResNet
        return ResNet()
    
    elif extractor_key == 'base':
        from MoEIR.modules.feature_extractors import BaseNet
        return BaseNet(feature_size=f_size)

    else:
        raise ValueError


def get_expert_module(expert_key, f_size, ex_f_size):
    if expert_key == 'fvdsr':
        from MoEIR.modules.experts import FVDSRNet
        return FVDSRNet(feature_size = f_size,
                        out_feature_size = ex_f_size)
    else:
        raise ValueError


def get_gate_module(gate_key, f_size, ex_f_size, num_experts):
    if gate_key == 'gmp':
        from MoEIR.modules.gates import GMP_GateNet
        return GMP_GateNet(in_feature_size = f_size,
                           out_feature_size = ex_f_size,
                           num_experts = num_experts)
    elif gate_key == 'gap':
        from MoEIR.modules.gates import GAP_GateNet
        return GAP_GateNet(in_feature_size = f_size,
                           out_feature_size = ex_f_size,
                           num_experts = num_experts)
    else:
        raise ValueError


def get_recon_module(recon_key, ex_f_size, num_experts):
    if recon_key == 'base':
        from MoEIR.modules.reconstructors import BaseNet
        return BaseNet(in_channels = ex_f_size,
                       out_channels = 3)
    elif recon_key == 'cwa':
        from MoEIR.modules.reconstructors import ChannelWiseAttentionNet
        return ChannelWiseAttentionNet(in_channels = ex_f_size,
                                       out_channels = 3,
                                       num_experts = num_experts)
    else:
        raise ValueError


def get_attention_module(attention_key):
    if attention_key == 'base':
        from MoEIR.modules.attentions import BaseNet
        return BaseNet()
    else:
        raise ValueError
