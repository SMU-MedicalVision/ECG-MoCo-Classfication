#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Jiewei Lai

import scipy
import random

import numpy as np
import neurokit2 as nk

# Drop out in frequency domain
class ECGFrequencyDropOut(object):
    def __init__(self, rate=0.3, default_len=7500):
        self.rate = rate
        self.default_len = default_len
        self.num_zeros = int(self.rate * self.default_len)

    def __call__(self, data):
        num_zeros = random.randint(0, self.num_zeros)
        zero_idxs = sorted(np.random.choice(np.arange(self.default_len), num_zeros, replace=False))
        data_dct = scipy.fft.dct(data.copy())
        data_dct[:, zero_idxs] = 0
        data_idct = scipy.fft.idct(data_dct)

        return data_idct

# Randomly select signals longer than n seconds and adjust them to 15s
class ECGCropResize(object):
    def __init__(self, n=2, default_len=7500, fs=500):
        self.min_len = n * fs
        self.default_len = default_len

    def __call__(self, data):
        crop_len = random.randint(self.min_len, self.default_len)
        crop_start = random.randint(0, self.default_len - crop_len)
        data_crop = data[:, crop_start:crop_start + crop_len]
        data_resize = np.empty_like(data)
        x = np.linspace(0, crop_len-1, crop_len)
        xnew = np.linspace(0, crop_len-1, self.default_len)
        for i in range(data.shape[0]):
            f = scipy.interpolate.interp1d(x, data_crop[i], kind='cubic')
            data_resize[i] = f(xnew)

        return data_resize

# Select a certain length signal in each heartbeat cycle and set it to zero
class ECGCycleMask(object):
    def __init__(self, rate=0.5, fs=500):
        self.rate = rate
        self.fs = fs

    def __call__(self, data):
        try:
            # Extract R-peaks locations
            _, rpeaks = nk.ecg_peaks(np.float32(data[1]), sampling_rate=self.fs)
            r_peaks = rpeaks['ECG_R_Peaks']
            if len(r_peaks) > 1:
                cycle_len = int(np.mean(np.diff(r_peaks)))
                cut_len = int(self.rate * cycle_len)
                cut_start = random.randint(0, cycle_len - cut_len)
                data_ = data.copy()
                for r_idx in r_peaks:
                    data_[:, r_idx + cut_start:r_idx + cut_start + cut_len] = 0
                return data_
            else:
                return data
        except:
            return data

# Randomly select less than the number of masks and set these channels to zero
class ECGChannelMask(object):
    def __init__(self, masks=6, default_channels=12):
        self.masks = masks
        self.channels = np.arange(default_channels)

    def __call__(self, data):
        masks = random.randint(1, self.masks)
        channels_mask = np.random.choice(self.channels, masks, replace=False)
        data_ = data.copy()
        for channel_mask in channels_mask:
            data_[channel_mask] = 0
        return data_
