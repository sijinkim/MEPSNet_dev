def prepare_modules(module_map, device):
    module_seq = []
    for module_key in module_map.keys():
        if module_key == 'feature_extractor':
            module = get_feature_extractor_module(module_map[module_key])
            module = module.to(device)

        elif module_key == 'experts':
            # Because we need the multiple experts,
            # module, here, should be a list of the experts.
            module = []
            for expert_key in module_map[module_key]:
                module.append(get_expert_module(expert_key).to(device))

        elif module_key == 'gate':
            module = get_gate_module(module_map[module_key]).to(device)

        elif module_key == 'reconstructor':
            module = get_recon_module(module_map[module_key]).to(device)

        elif module_key == 'attention':
            module = get_attention_module(module_map[module_key]).to(device)

        else:
            raise ValueError

        module_seq.append(module)

    return module_seq


def get_feature_extractor_module(extractor_key):
    if extractor_key == 'resnet':
        from MoEIR.modules.feature_extractors import ResNet
        return ResNet()
    else:
        raise ValueError


def get_expert_module(expert_key):
    if expert_key == 'fvdsr':
        from MoEIR.modules.experts import FVDSRNet
        return FVDSRNet()
    else:
        raise ValueError


def get_gate_module(gate_key):
    if gate_key == 'base':
        from MoEIR.modules.gates import BaseNet
        return BaseNet()
    else:
        raise ValueError


def get_recon_module(recon_key):
    if recon_key == 'base':
        from MoEIR.modules.reconstructors import BaseNet
        return BaseNet()
    elif recon_key == 'cwa':
        from MoEIR.modules.reconstructors import ChannelWiseAttentionNet
        return ChannelWiseAttentionNet()
    else:
        raise ValueError


def get_attention_module(attention_key):
    if attention_key == 'base':
        from MoEIR.modules.attentions import BaseNet
        return BaseNet()
    else:
        raise ValueError
