def prepare_modules(module_map, device):
    module_seq = []
    for module_key in module_map.keys():
        if module_key == 'feature_extractor':
            if module_map[module_key] == 'resnet':
                from MoEIR.modules.feature_extractors import ResNet
                module = ResNet()
            else:
                raise ValueError

        elif module_key == 'experts':
            print('experts:', module_map[module_key])
        elif module_key == 'reconstructor':
            print('reconstructor:', module_map[module_key])
        elif module_key == 'attention':
            print('attention:', module_map[module_key])

        module_seq.append(module)

    return module_seq