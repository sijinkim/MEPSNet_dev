import os
import argparse

from torch.utils.data import DataLoader
from MoEIR.data.dataset import TrainDataset, ValidDataset, TestDataset


parser = argparse.ArgumentParser(prog='MoEIR')
parser.add_argument('--n_partition', type=str, default='2', help='Number of data partition')
parser.add_argument('--patchsize', type=int, default=41)

parser.add_argument('--batchsize', type=int, default=128)

# TODO: YB: Prepare module sequence.

# TODO: SJ: Load datasets
opt = parser.parse_args()

train_dataset = TrainDataset(size = opt.patchsize, n_partition = opt.n_partition)
train_loader = DataLoader(train_dataset,
                          batch_size = opt.batchsize,
                          drop_last = True,
                          shuffle = True)

# TODO: SJ: Train

for index, (data, ref) in enumerate(train_loader):
    print(index)
    print('data:', data.shape)
    print('ref:', ref.shape)
    

