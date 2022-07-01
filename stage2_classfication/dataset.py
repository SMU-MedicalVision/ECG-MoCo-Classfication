#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Jiewei Lai

import xlrd
import torch

import numpy as np
from torch.utils.data import DataLoader

# Read the label code excel
def read_diagnostic_code_excel(CodeClassifyExcel):
    table = xlrd.open_workbook(CodeClassifyExcel)
    table = table.sheet_by_index(0)
    names = table.col_values(2)
    codes = table.col_values(3)
    terms = table.col_values(4)

    del names[0]
    del terms[0]
    del codes[0]

    assert len(names) == len(terms) and len(names) == len(codes)

    codes_dict1, codes_dict2 = dict(), dict()
    value = None
    for i, name in enumerate(names):
        if name != '':
            value = name
            codes_dict1[value] = list()
            codes_dict1[value].append(codes[i])
            codes_dict2[terms[i]] = codes[i]
        else:
            names[i] = value
            codes_dict1[value].append(codes[i])
            codes_dict2[terms[i]] = codes[i]

    return codes_dict1, codes_dict2

class ECGDataSet(torch.utils.data.Dataset):

    def __init__(self, ecgs, targets, data_idx, transform=None):
        self.ecgs = ecgs
        self.targets = targets
        self.data_idx = data_idx
        self.transform = transform

    def __getitem__(self, index):
        index = self.data_idx[index]
        ecg = self.ecgs[index]
        target = self.targets.loc[index].values
        if self.transform is not None:
            ecg = self.transform(ecg)
        return ecg.astype(np.float32), target.astype(np.float32)

    def __len__(self):
        assert len(self.ecgs) == len(self.targets)
        return len(self.data_idx)