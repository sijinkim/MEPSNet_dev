import os
import h5py
import glob
import random
import numpy as npe
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

        path = f'/home/tiwlsdi0306/workspace/image_dataset/DIV2K/part_distorted/DIV2K_part_distorted_train_part{n_partition}.h5'
        self.size = size        

        h5f = h5py.File(path, 'r')
        groups = list(h5f.keys())
        
        self.data_ = h5f.get(groups[0])
        self.target = h5f.get(groups[1])
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        size = self.size
        
        try:
            data_, target = random_crop(data = self.data_[str(index)][:,:,:], target = self.target[str(int(index//12))][:,:,:], size = size)
        except KeyError:
            print(f'[KeyError]index:{str(index)}, target index:{str(index//12)}')
            raise KeyError
        
        return self.transform(data_), self.transform(target)

    def __len__(self):
        return len(self.data_)


class ValidDataset(data.Dataset):
    def __init__(self, path='/home/tiwlsdi0306/image_dataset/DIV2K', part=2):
        super(ValidDataset, self).__init__()
        
        self.path = path
        self.part = part
        self.dataset = path.split('/')[-1] #DIV2K,Set5,Set14,Urban100, BSDS100
 
        self.data_ = glob.glob(os.path.join(self.path,'part_distorted', f'{self.dataset}_valid_HR', f'part{self.part}', '*.png'))
        self.target = glob.glob(os.path.join(self.path, 'reference', f'{self.dataset}_valid_HR', '*.png'))
        
        self.data_.sort()
        self.target.sort()

        self.data_ = self.data_[:60]
        self.target = self.target[:5]
        
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        data_ = Image.open(self.data_[index])
        target = Image.open(self.target[index//12])

        data_ = data_.convert('RGB')
        target = target.convert('RGB')
        filename = self.data_[index].split('/')[-1] #e.g. 0801_11.png

        return self.transform(data_), self.transform(target), filename
        
    def __len__(self):
        return len(self.data_)
  
         
class TestDataset(data.Dataset):
    def __init__(self, path='/home/tiwlsdi0306/image_dataset/DIV2K', part=2):
        super(TestDataset, self).__init__()
        
        self.path = path
        self.part = part
        self.dataset = path.split('/')[-1] #DIV2K,Set5,Set14,Urban100, BSDS100
 
        self.data_ = glob.glob(os.path.join(self.path,'part_distorted', f'{self.dataset}_test_HR', f'part{self.part}', '*.png'))
        self.target = glob.glob(os.path.join(self.path, 'reference', f'{self.dataset}_test_HR', '*.png'))
        
        self.data_.sort()
        self.target.sort()
        
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        data_ = Image.open(self.data_[index])
        target = Image.open(self.target[index//12])

        data_ = data_.convert('RGB')
        target = target.convert('RGB')
        filename = self.data_[index].split('/')[-1] #e.g. 0801_11.png

        return self.transform(data_), self.transform(target), filename
        
    def __len__(self):
        return len(self.data_)
 
