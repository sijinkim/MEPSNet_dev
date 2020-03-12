import argparse
import numpy as np
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim 

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader

from MoEIR.data import TrainDataset, ValidTestDataset
from MoEIR.measure import PSNR_SSIM_by_type

parser = argparse.ArgumentParser(prog='MoEIR')
#dataset setting
parser.add_argument('--dataset', type=str, default='DIV2K')
parser.add_argument('--n_noise', type=str, default='4', help='Number of noise - 4 or 6?')
parser.add_argument('--n_partition', type=str, default='2', help='Number of data partition - 2, 4, Mix')
parser.add_argument('--n_valimages', type=int, default=5, help='Number of images using in validation phase')
parser.add_argument('--datapath', type=str, default='/home/tiwlsdi0306/workspace/image_dataset/noiseInfo/')
parser.add_argument('--n_worker', type=int, default=1)

#train setting
parser.add_argument('--patchsize', type=int, default=41)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--featuresize', type=int, default=256)
parser.add_argument('--ex_featuresize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weightdecay', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--cpu', action='store_true', help="Use CPU only")
parser.add_argument('--val_term', type=int, default=10, help='Set saving models and validation test term')

#snapshot
parser.add_argument('--modelsave', action='store_true', help='True: save models, False: not save models')

#modules
parser.add_argument('--feature_extractor', type=str, default='base')
parser.add_argument('experts', type=str, nargs='+')
parser.add_argument('--kernelsize', type=int, nargs='*', help='Must match the length with the number of experts. (take 1, 3, 5, or 7)')
parser.add_argument('--gate', type=str, help='Take gmp or gap')
parser.add_argument('--reconstructor', type=str, default='ReconstructNet')
parser.add_argument('--attention', action='store_true', help='Use Attention method')
parser.add_argument('--multi_attention', action='store_true', help='Use GMP+GAP attention')
parser.add_argument('--gmp_k', type=int, default=10, help='number of GMP classes')

parser.add_argument('--comment', type=str, help='GATE or ATTENTION - using in writer(tebsorboard)')
opt = parser.parse_args()

print('Start setting')

if opt.gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{opt.gpu}')
    print(f'Using CUDA gpu{opt.gpu}')


#set tensorboardX writer
writer = SummaryWriter(log_dir=os.path.join(f'/home/tiwlsdi0306/workspace/MoEIR_compare_runs/part{opt.n_partition}', f'{opt.experts[0]}_{len(opt.experts)}_patch{opt.patchsize}_batch{opt.batchsize}_feature{opt.featuresize}_{opt.comment}'))

#set seed for train
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
print(f'Fix seed number: 0')

if opt.gate:
    from MoEIR.modules import MoE_with_Gate

    train_sequence = MoE_with_Gate(device=device,
                                   n_experts=len(opt.experts),
                                   args=opt)        



elif opt.attention:
    from MoEIR.modules import MoE_with_Attention
    
    train_sequence = MoE_with_Attention(device=device,
                                        n_experts=len(opt.experts),
                                        args=opt)
else:
    raise ValueError

module_sequence = train_sequence.take_modules()
print('Prepare module sequence')
module_sequence_keys = list(module_sequence.keys())
print(module_sequence_keys)
#module_sequence: dictionary
#{'feature_extractor': , 'experts':, '(gate) or (attention)': , reconstructor:}

#set checkpoint path
checkpoint_path = os.path.join('/home/tiwlsdi0306/workspace/snapshot/MoEIR_checkpoint', f'part{opt.n_partition}') 
PATH = os.path.join(checkpoint_path, f'{opt.experts[0]}_{len(opt.experts)}_{module_sequence_keys[2]}_{opt.comment}.tar')

train_dataset = TrainDataset(size=opt.patchsize, n_partition=opt.n_partition)
train_loader = DataLoader( train_dataset,
		                   batch_size=opt.batchsize,
		                   drop_last=True,
                           num_workers=opt.n_worker,
                           #pin_memory=not opt.cpu,
		                   shuffle=True)
print(f'Train dataset: part{opt.n_partition} distorted data - length of data: {len(train_dataset)}')

#Load validation data
valid_dataset = ValidTestDataset(dataset=opt.dataset, n_partition=opt.n_partition, num_images=opt.n_valimages, type_='valid')
valid_loader = DataLoader( valid_dataset,
		                   batch_size=1,
		                   drop_last=True,
                           num_workers=opt.n_worker,
                           #pin_memory=not opt.cpu,
		                   shuffle=True)

criterion = nn.MSELoss(reduction='sum')


optimizer = optim.Adam(
    [{'params':net.parameters() for net in module_sequence[module_sequence_keys[1]]}, #for experts
    {'params':net.parameters() for i,net in enumerate(module_sequence.values()) if i != 1}],
    weight_decay=opt.weightdecay,
    lr=opt.lr
)

scheduler = MultiStepLR(optimizer, milestones=[200, 500], gamma=0.5)
print('LOSS: MSE loss, optimizer: Adam, Using MultiStepLR scheduler[200,500]')

print("Start Training")
epoch = 1
loss_record = 0

while True:
    print(f"Epoch={epoch}, lr={optimizer.param_groups[0]['lr']}/ expert lr = {optimizer.param_groups[1]['lr']}")
    cost = 0

    for index, (data, ref) in enumerate(train_loader):
        optimizer.zero_grad()
    
        data = data.to(device)
        ref = ref.to(device)
        
        #Train
        outputs = train_sequence(data) 
        
        #Calculate loss
        loss = criterion(outputs, ref) #per batch

        print(f'Epoch[{epoch}/{index}] Ours Loss: {format(loss/opt.batchsize, ".3f")}')
        cost += loss #per epoch
        
        #back propagation
        loss.backward()
        optimizer.step()
    
#    writer.add_scalar(f'TRAIN/LOSS', cost/(len(train_dataset)//opt.batchsize), epoch) 
    writer.add_scalar(f'TRAIN/LOSS', cost/len(train_loader), epoch) 
    loss_record = loss



    #Validation 
    if epoch % opt.val_term == 0:
        #set measure
        measure = PSNR_SSIM_by_type(dataset=opt.dataset, num_partition=opt.n_partition, phase_type='valid')
        psnr_record = 0
        #ssim_record = 0

        print(f'[EPOCH{epoch}] Validation\n dataset: {opt.dataset} part{opt.n_partition} distorted data')
        
        with torch.no_grad():
            for step, (data, ref, filename) in enumerate(valid_loader):
                ref = ref.squeeze(0) #torch.Tensor [3, h, w]
                filename = str(filename)[2:-3]
                
                result_patch = []
                for patch_idx, patch in enumerate(data):
                    patch = patch.to(device).squeeze(0)
                    outputs = train_sequence.forward_valid_phase(patch).squeeze(0)
                    result_patch.append(outputs)
                #Merge 8 image patches
                h, w = ref.size()[1:]
                h_half, w_half = int(h/2), int(w/2)
                h_quarter, w_quarter = int(h_half/2), int(w_half/2)
                h_shave, w_shave = int(h_quarter/2), int(w_quarter/2)
                h_chop, w_chop = h_half + h_shave, w_quarter + w_shave
                
                for idx in range(len(result_patch)):
                    result_patch[idx] = result_patch[idx].cpu().mul(255).clamp(0,255).byte().permute(1,2,0).numpy() #h x w x 3

                result = np.ndarray(shape=(h, w, 3))
                result[0:h_half, 0:w_quarter, :] = result_patch[0][0:h_half, 0:w_quarter, :]
                result[0:h_half, w_quarter:w_half, :] = result_patch[1][0:h_half, 0:w_quarter, :]  
                result[0:h_half, w_half:w-w_quarter, :] = result_patch[2][0:h_half, 0:w_quarter, :]
                result[0:h_half, w-w_quarter:w, :] = result_patch[3][0:h_half, w_shave:w_chop, :]
                result[h_half:h, 0:w_quarter, :] = result_patch[4][h_shave:h_chop, 0:w_quarter, :]
                result[h_half:h, w_quarter:w_half, :] = result_patch[5][h_shave:h_chop, 0:w_quarter, :]
                result[h_half:h, w_half:w-w_quarter, :] = result_patch[6][h_shave:h_chop, 0:w_quarter, :]
                result[h_half:h, w-w_quarter:w, :] = result_patch[7][h_shave:h_chop, w_shave:w_chop, :]
                #Evaluate PSNR
                ref_array = ref.mul(255).clamp(0,255).byte().permute(1,2,0).numpy()

                min_val, max_val = 0, 255
                result_array = (result.astype(np.float64) - min_val) / (max_val-min_val)
                ref_array = (ref_array.astype(np.float64) - min_val) / (max_val-min_val)

                psnr_ = psnr(ref_array, result_array, data_range=1)
                #Evaluate SSIM
                #ssim_ = ssim(ref_array, result_array, data_range=1, multichannel=True)
                print(f"Epoch[{epoch}/{step}] Image {filename}, PSNR:{format(psnr_,'.3f')}") 
 
                psnr_record += psnr_
                #ssim_record += ssim_
                
                #Evaluate PSNR, SSIM by type
                measure.get_psnr(x=result_array, ref=ref_array, image_name=filename)
                #measure.get_ssim(x=result_array, ref=ref_array, image_name=filename) 


        writer.add_scalar(f'VALID/PSNR', psnr_record/len(valid_loader), epoch)
        #writer.add_scalar(f'VALID/SSIM', ssim_record/len(valid_loader), epoch)


        #psnr, ssim by type
        psnr_result = measure.get_psnr_result()
        #ssim_result = measure.get_ssim_result()
        writer.add_scalars(f'VALID/TYPE_PSNR', {'gwn':psnr_result['gwn'], 'gblur':psnr_result['gblur'], 'contrast':psnr_result['contrast'], 'fnoise':psnr_result['fnoise']}, epoch)
        #writer.add_scalars(f'VALID/TYPE_SSIM', {'gwn':ssim_result['gwn'], 'gblur':ssim_result['gblur'], 'contrast':ssim_result['contrast'], 'fnoise':ssim_result['fnoise']}, epoch)


        scheduler.step()      

        #model save if you want
        if opt.modelsave:
            
            torch.save({'feature_extractor_state_dict': module_sequence[module_sequence_keys[0]].state_dict(),
                        'experts_state_dict': [net.state_dict() for net in module_sequence[module_sequence_keys[1]]],
                        f'{module_sequence_keys[2]}_state_dict': module_sequence[module_sequence_keys[2]].state_dict(),
                        'reconstructor_state_dict': module_sequence[module_sequence_keys[3]].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss':loss_record ,
                        'epoch': epoch}, 
                        PATH)
            print('Model save: ', PATH)            
    epoch += 1
    


