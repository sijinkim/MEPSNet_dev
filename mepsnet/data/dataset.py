import h5py
import glob
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image


def random_crop(data, target, size):
    # data: distorted image
    # target: clean image(GT)
    # size: patch size for training

    h, w = data.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    crop_data = data[y:y+size, x:x+size].copy()
    crop_target = target[y:y+size, x:x+size].copy()

    return crop_data, crop_target


def ToTensorWithoutScaling(data):
    # data: numpy array image with range (0, 255)
    # transform numpy array image to tensor with range (0, 255)
    tensor_image = torch.FloatTensor(np.array(data)).permute(2, 0, 1)

    return tensor_image


class SHDD_train(data.Dataset):
    def __init__(self, size, level):
        super(SHDD_train, self).__init__()
        # size: train image patch size
        # level: level of SHDD according to the number of blocks in a single image(default: moderate)

        self.path = f'/data/DIV2K/train/DIV2K_{level}.h5'
        self.size = size

    def __getitem__(self, index):
        size = self.size

        with h5py.File(self.path, 'r') as db:
            groups = list(db.keys())
            group_0 = db.get(groups[0])
            group_1 = db.get(groups[1])

            try:
                data_, target = random_crop(
                    data=group_0[str(index)][:, :, :],
                    target=group_1[str(int(index//12))][:, :, :],
                    size=size)

            except KeyError:
                print(
                    f'[KeyError] data index:{str(index)}, target index:{str(index//12)}')
                raise KeyError

        return ToTensorWithoutScaling(data_), ToTensorWithoutScaling(target)

    def __len__(self):
        with h5py.File(self.path, 'r') as db:
            groups = list(db.keys())
            length = len(db[groups[0]])
        return length


class SHDD_test(data.Dataset):
    def __init__(self, level, num_images, type_):
        super(SHDD_test, self).__init__()
        # level: level of SHDD according to the number of blocks in a single image(default: moderate)
        # num_images: number of images to (validate, test) model
        # type_: 'valid' or 'test'

        self.data_ = glob.glob(f'/data/DIV2K/{type_}/DIV2K_{level}/*.png')
        self.target = glob.glob('/data/DIV2K/reference/*.png')
        self.data_.sort()
        self.target.sort()

        self.data_ = self.data_[:num_images * 12]
        self.target = self.target[:num_images]

    def __getitem__(self, index):
        data_ = Image.open(self.data_[index])
        target = Image.open(self.target[index//12])
        filename = self.data_[index].split('/')[-1]  # e.g. ('0801_11.png'),

        data_ = ToTensorWithoutScaling(data_)
        target = ToTensorWithoutScaling(target)

        h, w = data_.size()[1:]

        h_half, w_half = int(h/2), int(w/2)
        h_quarter, w_quarter = int(h_half/2), int(w_half/2)
        h_shave, w_shave = int(h_quarter/2), int(w_quarter/2)

        h_chop, w_chop = h_half + h_shave, w_quarter + w_shave

        input_patch_list = []
        input_patch_list.append(torch.FloatTensor(
            data_[:, 0:h_chop, 0:w_chop].unsqueeze(0)))
        input_patch_list.append(torch.FloatTensor(
            data_[:, 0:h_chop, w_quarter:w_half + w_shave].unsqueeze(0)))
        input_patch_list.append(torch.FloatTensor(
            data_[:, 0:h_chop, w_half:w_half + w_chop].unsqueeze(0)))
        input_patch_list.append(torch.FloatTensor(
            data_[:, 0:h_chop, w_half+w_quarter-w_shave:w].unsqueeze(0)))

        input_patch_list.append(torch.FloatTensor(
            data_[:, h_half-h_shave:h, 0:w_chop].unsqueeze(0)))
        input_patch_list.append(torch.FloatTensor(
            data_[:, h_half-h_shave:h, w_quarter:w_half + w_shave].unsqueeze(0)))
        input_patch_list.append(torch.FloatTensor(
            data_[:, h_half-h_shave:h, w_half:w_half + w_chop].unsqueeze(0)))
        input_patch_list.append(torch.FloatTensor(
            data_[:, h_half-h_shave:h, w_half+w_quarter-w_shave:w].unsqueeze(0)))

        return input_patch_list, target, filename

    def __len__(self):
        return len(self.data_)
