import argparse
import os

import torch
from torch.utils.data import DataLoader

from MoEIR.data.dataset import TrainDataset
from MoEIR.modules.utils import prepare_modules


parser = argparse.ArgumentParser(prog='MoEIR')
parser.add_argument('--n_partition', type=str, default='2', help='Number of data partition')
parser.add_argument('--patchsize', type=int, default=41)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--feature_extractor', type=str, default='resnet')
parser.add_argument('experts', type=str, nargs='+')
parser.add_argument('--gate', type=str, default='base')
parser.add_argument('--reconstructor', type=str, default='base')
parser.add_argument('--attention', type=str, default='base')
parser.add_argument('--gpu', type=int, default=None)

opt = parser.parse_args()

device = torch.device('cpu') if not opt.gpu \
    else torch.device(f'cuda:{opt.gpu}')


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

train_dataset = TrainDataset(size = opt.patchsize, n_partition = opt.n_partition)
train_loader = DataLoader(train_dataset,
                          batch_size = opt.batchsize,
                          drop_last = True,
                          shuffle = True)

for index, (data, ref) in enumerate(train_loader):
    print(index)
    print('data:', data.shape)
    print('ref:', ref.shape)
