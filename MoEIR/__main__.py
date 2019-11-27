import argparse

import torch

from MoEIR.modules.utils import prepare_modules

parser = argparse.ArgumentParser(prog='MoEIR')
parser.add_argument('--feature_extractor', type=str, default='resnet')
parser.add_argument('experts', type=str, nargs='+')
parser.add_argument('--gate', type=str, default='base')
parser.add_argument('--reconstructor', type=str, default='base')
parser.add_argument('--attention', type=str, default='base')
parser.add_argument('--gpu', type=int, default=None)

opt = parser.parse_args()

device = torch.device('cpu') if not opt.gpu \
    else torch.device(f'cuda:{opt.gpu}')

# TODO: YB: Prepare module sequence.
# Module preparation
module_sequence = prepare_modules(
    module_map={
        'feature_extractor': opt.feature_extractor,
        'experts': opt.experts,
        'gate': opt.gate,
        'reconstructor': opt.reconstructor,
        'attention': opt.attention,
    },
    device=device,
)

print(module_sequence)

# TODO: SJ: Load datasets

# TODO: SJ: Train
