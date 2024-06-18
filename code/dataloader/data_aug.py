import random
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F


np.seterr(divide='ignore', invalid='ignore')


class Normalization(object):
    def __init__(self, volume_key='volume2'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        arr = image_array.reshape(-1)
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        image_array = (image_array - arr_min) / (arr_max - arr_min + 1e-6)
        sample[self.volume_key] = image_array
        return sample


class RandomRotateTransform(object):
    def __init__(self, mask_prefix=False, box_prefix=False, angle_range=(-10, 10), p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.angle_range = angle_range
        self.mask_prefix = mask_prefix
        self.box_prefix = box_prefix

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if self.mask_prefix:
            mask = sample['mask']
        if self.box_prefix:
            box = sample['box']
        if np.random.uniform() < self.p_per_sample:
            rand_angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            volume1 = rotate(volume1, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
            volume2 = rotate(volume2, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
            if self.mask_prefix:
                mask = rotate(mask, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
            if self.box_prefix:
                box = rotate(box, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
        sample['volume1'] = volume1
        sample['volume2'] = volume2
        if self.mask_prefix:
            sample['mask'] = mask
        if self.box_prefix:
            sample['box'] = box
        return sample


class ScaleTransform(object):
    def __init__(self, zoom_range=(0.8, 1.3), p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.zoom_range = zoom_range

    def __call__(self, sample):
        volume, label = sample['volume'], sample['label']
        if np.random.uniform() < self.p_per_sample:
            zoom_factor = np.random.randint(self.zoom_range[0]*10, self.zoom_range[1]*10) / 10
            volume = zoom(volume, zoom_factor, order=1)
            label = zoom(label, zoom_factor, order=0)
        sample['volume'], sample['label'] = volume, label

        return sample


class MirrorTransform(object):
    def __init__(self, mask_prefix=False, box_prefix=False, axes=(0, 1, 2)):
        self.axes = axes
        self.mask_prefix = mask_prefix
        self.box_prefix = box_prefix

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if self.mask_prefix:
            mask = sample['mask']
        if self.box_prefix:
            box = sample['box']
        if isinstance(self.axes, int):
            if np.random.uniform() < 0.5:
                volume1 = np.flip(volume1, self.axes)
                volume2 = np.flip(volume2, self.axes)
                if self.mask_prefix:
                    mask = np.flip(mask, self.axes)
                if self.box_prefix:
                    box = np.flip(box, self.axes)
        else:
            for axis in self.axes:
                if np.random.uniform() < 0.5:
                    volume1 = np.flip(volume1, axis=axis)
                    volume2 = np.flip(volume2, axis=axis)
                    if self.mask_prefix:
                        mask = np.flip(mask, axis=axis)
                    if self.box_prefix:
                        box = np.flip(box, axis=axis)
        sample['volume1'] = volume1
        sample['volume2'] = volume2
        if self.mask_prefix:
            sample['mask'] = mask
        if self.box_prefix:
            sample['box'] = box
        return sample


class GaussianBlur(object):
    def __init__(self, sigma=3.0):
        self.sigma = sigma

    def __call__(self, sample):
        array = sample['mask']
        sample['mask'] = gaussian_filter(array, sigma=self.sigma)
        return sample
