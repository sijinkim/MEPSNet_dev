import argparse
import imageio
import numpy as np
import os
import sys
import math

import torch
from torch.utils.data import DataLoader

from MoEIR.data import ValidTestDataset
from MoEIR.measure import utility
from MoEIR.measure.psnr_ssim_by_type import PSNR_SSIM_by_type

parser = argparse.ArgumentParser()

# Dataset setting
parser.add_argument('--dataset', type=str, default='DIV2K', choices=['DIV2K', 'Set5', 'Set14', 'Urban100'])
parser.add_argument('--n_noise', type=str, default='4')
parser.add_argument('--data_partition', type=str, default='4', choices=['2', '4', 'Mix'])
parser.add_argument('--noiseinfo', type=str, default='/home/tiwlsdi0306/workspace/image_dataset/noiseInfo')
parser.add_argument('--n_images', type=int, default=50)

opt = parser.parse_args()

data_path = f'/home/tiwlsdi0306/workspace/image_dataset/{opt.dataset}/part_distorted/{opt.dataset}_test_HR'

print(f"Calculate psnr and ssim of noisy test data: {opt.dataset}_part{opt.data_partition}")
sys.stdout = open(f'./{opt.dataset}_part{opt.data_partition}_test.txt', 'w')
print(f'{opt}\n')

print(f"Calculate psnr and ssim of noisy test data: {opt.dataset}_part{opt.data_partition}")

test_dataset = ValidTestDataset(dataset=opt.dataset, n_partition=opt.data_partition, type_='test', num_images=opt.n_images)
test_loader = DataLoader(test_dataset,
                        batch_size=1,
                        drop_last=False,
                        num_workers=1,
                        shuffle=False)
inf_file = []
inf_check = 0
PSNR, SSIM = 0, 0
type_measure = PSNR_SSIM_by_type(dataset=opt.dataset, num_partition=opt.data_partition, phase_type='test')

for step, (data, ref, filename) in enumerate(test_loader):
    ref = ref.squeeze(0)
    filename = str(filename)[2:-3]

##########################################
    result_patch = []
    for patch_idx, patch in enumerate(data):
        result_patch.append(patch.squeeze(0).squeeze(0).cpu().clamp(0,255).round().permute(1,2,0).numpy())
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

    ref_array = ref.cpu().permute(1,2,0).numpy().astype(np.uint8)
    data_array = result.astype(np.uint8)

    psnr_ = utility.calc_psnr(ref_array, data_array)
    ssim_ = utility.calc_ssim(ref_array, data_array)
    print(f'{filename} PSNR: {psnr_}, SSIM: {ssim_}')

    if not psnr_ == float('inf'): 
        PSNR += psnr_
        SSIM += ssim_
    elif psnr_ == float('inf'):
        inf_check += 1
        inf_file.append(filename)
        print('PASS PSNR+, SSIM+')
    else:
        raise ValueError

    print(f'after add PSNR: {PSNR}, SSIM:{SSIM}')

    print('[TYPE PSNR]')
    type_measure.get_psnr(x=data_array, ref=ref_array, image_name=filename)
    print('[TYPE SSIM]')
    type_measure.get_ssim(x=data_array, ref=ref_array, image_name=filename)
    print('\n')


print(f'Inf files: {inf_file}... {inf_check} images')
print(f'[{opt.dataset}-part{opt.data_partition} Noisy images] Avg PSNR:{PSNR/(len(test_loader)-inf_check)}, SSIM: {SSIM/(len(test_loader)-inf_check)}')

TYPE_PSNR = type_measure.get_psnr_result()
TYPE_SSIM = type_measure.get_ssim_result()

print(f'Avg TYPE PSNR: {TYPE_PSNR}\nAvg TYPE SSIM: {TYPE_SSIM}')

sys.stdout.close()
