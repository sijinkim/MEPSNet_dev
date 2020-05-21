import argparse
import imageio
import numpy as np
import os
import sys
import random

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

from MoEIR.data import ValidTestDataset
from MoEIR.measure import utility
from MoEIR.measure.psnr_ssim_by_type import PSNR_SSIM_by_type

parser = argparse.ArgumentParser(prog='Test multi-noise image denoising')

# setting
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--n_worker', type=int, default=1)

# modules
parser.add_argument('--attention', action='store_true', help='The base module name is AttentionNet')
parser.add_argument('--gate', action='store_true')

# dataset
parser.add_argument('--dataset', type=str, default='DIV2K', choices=['DIV2K','Set5','Set14','Urban100'])
parser.add_argument('--n_noise', type=str, default='4')
parser.add_argument('--data_partition', type=str, default='4', choices=['2', '4','Mix'])
parser.add_argument('--datapath', type=str, default='/home/tiwlsdi0306/workspace/image_dataset/noiseInfo', help='datapath/dataset(Set5, Set14, Urban100, DIV2K)/{opt.dataset}_test_part{opt.data_partition}')
parser.add_argument('--n_images', type=int)

# models
parser.add_argument('experts', type=str, nargs='+')
parser.add_argument('--kernelsize', type=int, nargs='*')
parser.add_argument('--patchsize', type=int, default=48)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--featuresize', type=int, default=256)
parser.add_argument('--ex_featuresize', type=int, default=64)
parser.add_argument('--model_partition', type=str, default=4, choices=['2','4','Mix'])

parser.add_argument('--no_attention', action='store_true')
parser.add_argument('--n_bank', type=int, default=1)
parser.add_argument('--rir', action='store_true')
parser.add_argument('--rir_attention', action='store_true')
parser.add_argument('--n_sres', type=int, default=3)
parser.add_argument('--n_resblock', type=int, default=9)
parser.add_argument('--n_template', type=int, default=16)
parser.add_argument('--is_dilate', type=int, default=1, choices=[1,2,3])
parser.add_argument('--RIRintoBlock', action='store_true')
parser.add_argument('--cwa_fusion', action='store_true')
parser.add_argument('--conv_fusion', action='store_true')


parser.add_argument('--comment', type=str, default='Ntemplate16_AddInit_Resblock9*3_CWAinRIR')


parser.add_argument('--lite_feature', action='store_true')
parser.add_argument('--lite_reconst', action='store_true')
parser.add_argument('--multi-attention', action='store_true')

opt = parser.parse_args()


model_path = f'{opt.experts[0]}_{len(opt.experts)}_patch{opt.patchsize}_batch{opt.batchsize}_feature{opt.ex_featuresize}_{opt.comment}'

print(f'Test model path: part{opt.model_partition} trained {model_path}\n')
print(f'Test Dataset: {opt.dataset}-part{opt.data_partition}\n')

log_path = f'/home/tiwlsdi0306/workspace/snapshot/MoEIR_test/{opt.dataset}/part{opt.data_partition}/training_part{opt.model_partition}'
try: os.makedirs(log_path)
except FileExistsError: pass

sys.stdout = open(f'/home/tiwlsdi0306/workspace/snapshot/MoEIR_test/{opt.dataset}/part{opt.data_partition}/training_part{opt.model_partition}/{model_path}.txt', 'w')

print(f'options: {opt}\n')
print(f'Test model path: part{opt.model_partition} trained {model_path}\n')
print(f'Test Dataset: {opt.dataset}-part{opt.data_partition}\n')

print('[Start setting]\n')
if opt.gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{opt.gpu}')
    print(f'Using CUDA gpu{opt.gpu}') 


# random initialization
torch.manual_seed(0)
#np.random.seed(random.random())
#torch.cuda.manual_seed(random.random())

if opt.gate:
    from MoEIR.modules import MoE_with_Gate
    model = MoE_with_Gate()

elif opt.attention:
#    if not opt.no_attention and not opt.rir_attention: #RIR
#        from MoEIR.modules import MoE_with_Template
#        model = MoE_with_Template(device=device,
##                                     n_experts=len(opt.experts),
 #                                    args=opt)
    if opt.no_attention:
        from MoEIR.modules import MoE_with_Template_without_CWA #RIR without CWA
        model = MoE_with_Template_without_CWA(device=device, n_experts=len(opt.experts), args=opt)
    elif opt.rir:
        from MoEIR.modules import MoE_with_Template_CWA_in_RIR #CWAinRIR
        model = MoE_with_Template_CWA_in_RIR(device=device, n_experts=len(opt.experts), args=opt)
    else:
        raise ValueError
else:
    raise ValueError

# Prepare module_sequence
module_sequence = model.take_modules()
module_sequence_keys = list(module_sequence.keys())
print('Test model sequence: ', module_sequence)


# Read saved model
checkpoint_dir = f'/home/tiwlsdi0306/workspace/snapshot/MoEIR_checkpoint/part{opt.model_partition}'
#FILE = f'{opt.experts[0]}_{len(opt.experts)}_{module_sequence_keys[2]}_{opt.comment}.tar'

checkpoint = torch.load(os.path.join(checkpoint_dir, model_path + '.tar'))
print(f'Load model: {os.path.join(checkpoint_dir, model_path + ".tar")}')


# Load Feature extractor state
module_sequence[module_sequence_keys[0]].load_state_dict(checkpoint[f'{module_sequence_keys[0]}_state_dict'])
# Load Attention state
module_sequence[module_sequence_keys[2]].load_state_dict(checkpoint[f'{module_sequence_keys[2]}_state_dict'])
# Load Reconstructor state
module_sequence[module_sequence_keys[3]].load_state_dict(checkpoint[f'{module_sequence_keys[3]}_state_dict'])
for idx, state in enumerate(checkpoint[f'{module_sequence_keys[1]}_state_dict']):
    module_sequence[module_sequence_keys[1]][idx].load_state_dict(state)

# ValidTestDataset(type_='test')
test_dataset = ValidTestDataset(dataset=opt.dataset, n_partition=opt.data_partition, type_='test', num_images=opt.n_images)
test_loader = DataLoader(test_dataset,
                        batch_size=1,
                        drop_last=False,
                        num_workers=opt.n_worker,
                        shuffle=False)
# Evaluation
PSNR, SSIM  = 0, 0
type_measure = PSNR_SSIM_by_type(dataset=opt.dataset, num_partition=opt.data_partition, phase_type='test') 

with torch.no_grad():
        for step, (data, ref, filename) in enumerate(test_loader):
            ref = ref.squeeze(0) #torch.Tensor [3, h, w]
            filename = str(filename)[2:-3]
                
            result_patch = []
            for patch_idx, patch in enumerate(data):
                patch = patch.to(device).squeeze(0)
                outputs = model.forward(patch).squeeze(0).cpu().clamp(0,255).round().permute(1,2,0).numpy() #[0,255] range (H, W, 3) numpy array
                result_patch.append(outputs)
            #Merge 8 image patches
            h, w = ref.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_quarter, w_quarter = int(h_half/2), int(w_half/2)
            h_shave, w_shave = int(h_quarter/2), int(w_quarter/2)
            h_chop, w_chop = h_half + h_shave, w_quarter + w_shave
                
            result = np.ndarray(shape=(h, w, 3))
            result[0:h_half, 0:w_quarter, :] = result_patch[0][0:h_half, 0:w_quarter, :]
            result[0:h_half, w_quarter:w_half, :] = result_patch[1][0:h_half, 0:-w_shave, :]  
            result[0:h_half, w_half:w_half+w_quarter, :] = result_patch[2][0:h_half, 0:w_quarter, :]
            result[0:h_half, w_half+w_quarter:w, :] = result_patch[3][0:h_half, w_shave:, :]
            result[h_half:h, 0:w_quarter, :] = result_patch[4][h_shave:, 0:w_quarter, :]
            result[h_half:h, w_quarter:w_half, :] = result_patch[5][h_shave:, 0:-w_shave, :]
            result[h_half:h, w_half:w_half+w_quarter, :] = result_patch[6][h_shave:, 0:w_quarter, :]
            result[h_half:h, w_half+w_quarter:w, :] = result_patch[7][h_shave:, w_shave:, :]

            #Evaluate PSNR
            ref_array = ref.cpu().permute(1,2,0).numpy().astype(np.uint8)
            result_array = result.astype(np.uint8)
            
            psnr_ = utility.calc_psnr(ref_array, result_array)
            ssim_ = utility.calc_ssim(ref_array, result_array)
            print(f'{filename} PSNR: {psnr_}, SSIM: {ssim_}')
            PSNR += psnr_
            SSIM += ssim_
            
            # Evaluate by type of the noise
            print(f'[TYPE PSNR]')
            type_measure.get_psnr(x=result_array, ref=ref_array, image_name=filename) # Calculate per psnr of noises in each images
            print(f'[TYPE SSIM]')
            type_measure.get_ssim(x=result_array, ref=ref_array, image_name=filename)
            print('\n') 

            # save reconstructed image
            save_path = os.path.join(f'/home/tiwlsdi0306/workspace/snapshot/MoEIR_test/{opt.dataset}/part{opt.data_partition}', f'training_part{opt.model_partition}' ,model_path)
            try:
                os.makedirs(save_path)
            except FileExistsError:
                pass
            imageio.imwrite(os.path.join(save_path, f'{filename}'), result_array)

print(f'MODEL [{model_path}]\nAvg PSNR: {PSNR/len(test_loader)}, SSIM: {SSIM/len(test_loader)}')

TYPE_PSNR = type_measure.get_psnr_result()
TYPE_SSIM = type_measure.get_ssim_result()

print(f'Avg TYPE PSNR: {TYPE_PSNR}\nAvg TYPE SSIM: {TYPE_SSIM}')

sys.stdout.close()
