import os
import h5py
import glob
import random
import numpy as npe
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


def random_crop(data, target, size=41):
    h, w = data.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    crop_data = data[y:y+size, x:x+size].copy()
    crop_target = target[y:y+size, x:x+size].copy()

    return crop_data, crop_target

class TrainDataset(data.Dataset):
    def __init__(self, size, n_partition):
        super(TrainDataset, self).__init__()

        self.path = f'/home/tiwlsdi0306/workspace/image_dataset/DIV2K/part_distorted/DIV2K_part_distorted_train_part{n_partition}.h5'
        self.size = size        
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        size = self.size
        
        with h5py.File(self.path, 'r') as db:
            groups = list(db.keys())
            group_0 = db.get(groups[0])
            group_1 = db.get(groups[1])

            try:
                data_, target = random_crop(data = group_0[str(index)][:,:,:], target = group_1[str(int(index//12))][:,:,:], size = size)
            except KeyError:
                print(f'[KeyError]index:{str(index)}, target index:{str(index//12)}')
                raise KeyError
        
        return self.transform(data_), self.transform(target)

    def __len__(self):
        with h5py.File(self.path, 'r') as db:
            groups = list(db.keys())
            length = len(db[groups[0]])
        return length

class ValidTestDataset(data.Dataset):
    def __init__(self, dataset='DIV2K', n_partition=2, num_noise=4, num_images=5, type_='valid'):
        super(ValidTestDataset, self).__init__()
        
        self.path = os.path.join('/home/tiwlsdi0306/workspace/image_dataset', f'{dataset}')
        self.part = n_partition
        #dataset: DIV2K, Set5, Set14, Urban100, BSDS100

        if num_noise == 4:
            noise_type = 'part_distorted'

        elif num_noise == 6:
            noise_type = 'distorted'

        else:
            raise ValueError

        self.data_ = glob.glob(os.path.join(self.path, noise_type, f'{dataset}_{type_}_HR', f'part{self.part}', '*.png'))
        self.target = glob.glob(os.path.join(self.path, 'reference', f'{dataset}_{type_}_HR', '*.png'))
        self.data_.sort()
        self.target.sort()

        self.data_ = self.data_[:num_images * 12]
        self.target = self.target[:num_images]
        
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        data_ = Image.open(self.data_[index])
        target = Image.open(self.target[index//12])
        filename = self.data_[index].split('/')[-1] #e.g. ('0801_11.png'),
        
        data_ = self.transform(data_.convert('RGB'))
        target = self.transform(target.convert('RGB'))
        
        h, w = data_.size()[1:]
	
        h_half, w_half = int(h/2), int(w/2)
        h_quarter, w_quarter = int(h_half/2), int(w_half/2)
        h_shave, w_shave = int(h_quarter/2), int(w_quarter/2)
		
        h_chop, w_chop = h_half + h_shave, w_quarter + w_shave

        input_patch = self.make_patch_list(h=h_chop, w=w_chop)

        input_patch[0].copy_(data_[:, 0:h_chop, 0:w_chop])
        input_patch[1].copy_(data_[:, 0:h_chop, w_quarter:w_quarter + w_chop])
        input_patch[2].copy_(data_[:, 0:h_chop, w_half:w_half + w_chop])
        input_patch[3].copy_(data_[:, 0:h_chop, w-w_chop:w])
        input_patch[4].copy_(data_[:, h-h_chop:h, 0:w_chop])
        input_patch[5].copy_(data_[:, h-h_chop:h, w_quarter:w_quarter + w_chop])
        input_patch[6].copy_(data_[:, h-h_chop:h, w_half: w_half + w_chop])
        input_patch[7].copy_(data_[:, h-h_chop:h, w-w_chop:w])

        return input_patch, target, filename
        
    def __len__(self):
        return len(self.data_)

    def make_patch_list(self, h, w):
        patch = [torch.FloatTensor(1, 3, h, w) for _ in range(0,8)]
        return patch
 
