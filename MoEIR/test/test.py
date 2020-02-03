import argparse
import numpy as np
import os
import random

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from tensorboardX import SummaryWriter
import torch

parser = argparse.ArgumentParser(prog='test of MoEIR')

#setting
parser.add_argument('--gpu', type=int, default=None)

#modules
parser.add_argument('--attention', type=str, help='The base module name is AttentionNet')
parser.add_argument('--gate', type=str, help='Tamke gmp or gap')

opt = parser.parse_args()
writer = SummaryWriter(log_dir=f'/home/tiwlsdi0306/workspace/...')

print('Start setting')
if opt.gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{opt.gpu}')
    print(f'Using CUDA gpu{opt.gpu}') 


#random initialization
torch.manual_seed(random.random())
np.random.seed(random.random())
torch.cuda.manual_seed(random.random())

if opt.gate:
    from MoEIR.modules import MoE_with_Gate
    sequence = MoE_with_Gate()

elif opt.attention:
    from MoEIR.modules import MoE_with_Attention
    sequence = MoE_with_Attention()

else:
    raise ValueError

#prepare module_sequence
module_sequence = sequence.take_modules()
module_sequence_keys = list(module_sequence.keys())
print('Test model sequence: ', module_sequence)

#Read saved model
PATH = f'/home/tiwlsdi0306/workspace/snapshot/MoEIR_checkpoint/part{opt.n_partition}'
FILE = f'{opt.experts[0]}_{len(opt.experts)_{module_sequence_keys[2]}_{opt.comment}.tar}'

checkpoint = torch.load(os.path.join(PATH, FILE))

module_sequence[module_sequence_keys[0]].load_state_dict(checkpoint[f'{module_sequence_keys[0]}_state_dict'])
#for module_sequence[module_sequence_keys[1]].load_state_dict(checkpoint[f'{module_sequence_keys[0]}_state_dict'])
module_sequence[module_sequence_keys[2]].load_state_dict(checkpoint[f'{module_sequence_keys[2]}_state_dict'])
module_sequence[module_sequence_keys[3]].load_state_dict(checkpoint[f'{module_sequence_keys[3]}_state_dict'])

#ValidTestDaataset(type_='test')
test_dataset = ValidTestDataset(dataset=opt.dataset, n_partition=opt.n_partition, type_ = 'test')
test_loader = DataLoader(test_dataset,
                        batch_size=1,
                        drop_last=False,
                        num_workers=opt.n_worker,
                        shuffle=True)

#TODO: Adding save result(3 channels image) into snapshot/MoEIR_result/part2. part4. partMix

