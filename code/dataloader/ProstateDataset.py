import torch
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import SimpleITK as sitk


class PD1C_B_E_gary(Dataset):
    def __init__(self, case_list, b_root_path, e_root_path, transform=None):
        self.case_list = case_list
        self.b_root_path = b_root_path
        self.e_root_path = e_root_path
        self.transform = transform

    def __getitem__(self, index):
        data_path = os.path.join(self.b_root_path, self.case_list[index] + '.npz')
        swe_path = os.path.join(self.e_root_path, self.case_list[index] + '.npy')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        volume = data['volume1']
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume1 = volume.transpose(2, 1, 0)
        # volume = volume[0:1, :, :, :]
        swe = np.load(swe_path).transpose(2, 1, 0)
        # swe = np.expand_dims(swe, axis=0)
        name = self.case_list[index]
        mask = data['mask'].transpose(2, 1, 0)
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': name, 'volume1': volume1, 'volume2': swe, 'mask': mask,
                  'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PD1C_B_E_gary_bbox(Dataset):
    def __init__(self, case_list, b_root_path, e_root_path, box_root_path, transform=None):
        self.case_list = case_list
        self.b_root_path = b_root_path
        self.e_root_path = e_root_path
        self.box_root_path = box_root_path
        self.transform = transform

    def __getitem__(self, index):
        data_path = os.path.join(self.b_root_path, self.case_list[index] + '.npz')
        swe_path = os.path.join(self.e_root_path, self.case_list[index] + '.npy')
        box_path = os.path.join(self.box_root_path, self.case_list[index] + '.npy')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        volume = data['volume1']
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume1 = volume.transpose(2, 1, 0)
        # volume = volume[0:1, :, :, :]
        swe = np.load(swe_path).transpose(2, 1, 0)
        box = np.load(box_path).transpose(2, 1, 0).astype(np.int8)
        # swe = np.expand_dims(swe, axis=0)
        name = self.case_list[index]
        mask = data['mask'].transpose(2, 1, 0)
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': name, 'volume1': volume1, 'volume2': swe, 'mask': mask, 'box': box,
                  'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PDCAM(Dataset):
    def __init__(self, case_list, data_root_path, image_id='', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.transform = transform
        self.image_id = image_id

    def __getitem__(self, index):
        data_path = os.path.join(self.data_root_path, self.case_list[index] + '.npz')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        volume = data['volume{}'.format(self.image_id)]
        benign_malignant = data['label']
        # c z y x -> x y z
        volume = volume[:, :, :, :].transpose(0, 3, 2, 1)
        # volume = volume[0:1, :, :, :]
        name = self.case_list[index]
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': case_id, 'volume': volume, 'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class NormaKScore(object):
    def __init__(self, volume_key='volume'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        arr = image_array.reshape(-1)
        arr_mean = np.mean(arr)
        arr_var = np.var(arr)
        image_array = (image_array - arr_mean) / (arr_var + 1e-6)
        sample[self.volume_key] = image_array
        return sample


class ToTensor(object):
    def __init__(self, mask_prefix=False, box_prefix=False, channels=0):
        self.channels = channels
        self.mask_prefix = mask_prefix
        self.box_prefix = box_prefix

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if self.mask_prefix:
            mask = np.expand_dims(sample['mask'], axis=self.channels)
        if self.box_prefix:
            box = np.expand_dims(sample['box'], axis=self.channels)
        volume1 = np.expand_dims(volume1, axis=self.channels)
        volume2 = np.expand_dims(volume2, axis=self.channels)
        sample['volume1'] = torch.from_numpy(volume1.copy())
        sample['volume2'] = torch.from_numpy(volume2.copy())
        if self.mask_prefix:
            sample['mask'] = torch.from_numpy(mask.copy())
        if self.box_prefix:
            sample['box'] = torch.from_numpy(box.copy())
        return sample


class SparseZSlice(object):
    def __init__(self, sample_interval=2, volume_key='volume'):
        self.volume_key = volume_key
        self.sample_interval = sample_interval

    def __call__(self, sample):
        volume = sample[self.volume_key]
        z_slice = volume.shape[-1]
        if z_slice % self.sample_interval != 0:
            z_num = z_slice - (z_slice % self.sample_interval)
            volume = volume[:, : z_num, ...]
        prob = np.random.random()
        if self.sample_interval == 3:
            if prob < 1/3:
                volume = volume[..., ::self.sample_interval]
            elif 1/3 < prob < 2/3:
                volume = volume[..., 1::self.sample_interval]
            else:
                volume = volume[..., 2::self.sample_interval]
        elif self.sample_interval == 2:
            if prob < 0.5:
                volume = volume[..., ::self.sample_interval]
            else:
                volume = volume[..., 1::self.sample_interval]
        elif self.sample_interval == 4:
            if prob < 0.25:
                volume = volume[..., ::self.sample_interval]
            elif 0.25 < prob < 0.5:
                volume = volume[..., 1::self.sample_interval]
            elif 0.5 < prob < 0.75:
                volume = volume[..., 2::self.sample_interval]
            else:
                volume = volume[..., 3::self.sample_interval]
        else:
            return -1
        sample[self.volume_key] = volume

        return sample


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def next(self):
        sample = self.sample
        self.preload()
        return sample

    def preload(self):
        try:
            self.sample = next(self.loader)
        except StopIteration:
            self.sample = None
            return



