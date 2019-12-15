import argparse
import numpy as np
import os

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from MoEIR.data.dataset import TrainDataset

parser = argparse.ArgumentParser(prog='MoEIR')
#dataset setting
parser.add_argument('--dataset', type=str, default='DIV2K')
parser.add_argument('--n_noise', type=str, default='4', help='Number of noise - 4 or 6?')
parser.add_argument('--n_partition', type=str, default='2', help='Number of data partition')

#train setting
parser.add_argument('--patchsize', type=int, default=41)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--featuresize', type=int, default=256)
parser.add_argument('--ex_featuresize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weightdecay', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=None)


#modules
parser.add_argument('--feature_extractor', type=str, default='base')
parser.add_argument('experts', type=str, nargs='+')
parser.add_argument('--gate', type=str, help='Take gmp or gap')
parser.add_argument('--reconstructor', type=str, default='base')
parser.add_argument('--attention', type=str, help='The base is AttentionNet')

opt = parser.parse_args()

print('Start setting')
device = torch.device('cpu') if not opt.gpu \
else torch.device(f'cuda:{opt.gpu}')
print(f'Using CUDA gpu{opt.gpu}')


#set seed for train
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
print(f'Fix seed number: 0')

writer = SummaryWriter(log_dir=f'/home/tiwlsdi0306/workspace/runs/noise{opt.n_noise}/part{opt.n_partition}')

if opt.gate:
    from MoEIR.modules.experts_gate import MoE_with_Gate

    train_sequence = MoE_with_Gate(device=device,
                                   feature_size=opt.featuresize,
                                   expert_feature_size=opt.ex_featuresize,
                                   gate=opt.gate,
                                   n_experts=len(opt.experts),
                                   batch_size=opt.batchsize)
elif opt.attention:
    from MoEIR.modules.experts_attention import MoE_with_Attention
    
    train_sequence = MoE_with_Attention(device=device,
                                        feature_size=opt.featuresize,
                                        expert_feature_size=opt.ex_featuresize,
                                        n_experts=len(opt.experts),
                                        batch_size=opt.batchsize)
else:
    raise ValueError

module_sequence = train_sequence.take_modules()
print('Prepare module sequence')
print(module_sequence)


train_dataset = TrainDataset(size=opt.patchsize, n_partition=opt.n_partition)
train_loader = DataLoader(
                  train_dataset,
		  batch_size=opt.batchsize,
		  drop_last=True,
		  shuffle=True)
print(f'Train dataset: part{opt.n_partition} distorted data')

criterion = nn.MSELoss(size_average=False)
optimizer = optim.Adam(
    [{'params':net.parameters() for net in module_sequence[1]},
    {'params':net.parameters() for i,net in enumerate(module_sequence) if i != 1}],
    weight_decay=opt.weightdecay,
    lr=opt.lr
)
scheduler = ReduceLROnPlateau(
    optimizer=optimizer, 
    mode='min',
    factor=0.1, 
    verbose=True
)
print('LOSS: MSE loss, optimizer: Adam, Using scheduler')

print("Start Training")
epoch = 1

while True:
    print(f"Epoch={epoch}, lr={optimizer.param_groups[0]['lr']}/ expert lr = {optimizer.param_groups[1]['lr']}")
    cost = 0

    for index, (data, ref) in enumerate(train_loader):
        optimizer.zero_grad()
    
        data = data.to(device)
        ref = ref.to(device)
        
        outputs = train_sequence(data) 
        #Calculate loss
        loss = criterion(outputs, ref).div(opt.batchsize)

        print(f'Epoch[{epoch}/{index}] Ours Loss: {loss}')
        cost += loss
        
        #back propagation
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        mean_psnr, mean_ssim, mean_lpips = 0
        mean_loss, loss_record = 0
        
        #Load validation data
        valid_dataset = ValidDataset(dataset=opt.dataset, n_partition=opt.n_partition)
        valid_loader = DataLoader(
                  valid_dataset,
		  batch_size=1,
		  drop_last=True,
		  shuffle=True)
        print(f'[EPOCH{epoch}] Validation\n dataset: part{opt.n_partition} distorted data')
    
        #Image cropping: crop images into smaller size
        for step, (data, ref) in enumerate(valid_loader):
            print(f'step {step}')
            data = data.squeeze(0)
            ref = ref.squeeze(0)
            h,w = data.size()[1:] #data: [c, h ,w]
            print(f'h, w: {h} {w}')

            h_half, w_half = int(h/2), int(w/2)
            h_quarter, w_quarter = int(h_half/2), int(w_half/2)
            h_shave, w_shave = int(h_quarter/2), int(w_quarter/2)
            h_chop, w_chop = h_half + h_shave, w_quarter + w_shave
 
            #split whole image into 8 patches
            patch1 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch2 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch3 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch4 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch5 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch6 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch7 = torch.FloatTensor(1, 3, h_chop, w_chop)
            patch8 = torch.FloatTensor(1, 3, h_chop, w_chop)

            patch1.copy_(data[:, 0:h_chop, 0:w_chop])
            patch2.copy_(data[:, 0:h_chop, w_quarter:w_quarter+w_chop])
            patch3.copy_(data[:, 0:h_chop, w_half:w_half+w_chop])
            patch4.copy_(data[:, 0:h_chop, w-w_chop:w])




       


    epoch += 1
    

    
