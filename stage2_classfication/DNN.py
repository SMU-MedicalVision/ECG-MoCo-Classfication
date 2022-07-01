#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import torch
import torch.nn as nn

class SELayer1D(nn.Module):

    def __init__(self, nChannels, reduction=16):
        super(SELayer1D, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.se_block = nn.Sequential(
            nn.Linear(nChannels, nChannels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChannels // reduction, nChannels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        alpha = torch.squeeze(self.globalavgpool(x))
        alpha = self.se_block(alpha)
        alpha = torch.unsqueeze(alpha, -1)
        out = torch.mul(x, alpha)
        return out

class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, drop_out_rate, stride):
        super(BasicBlock1D, self).__init__()
        self.operation = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_out_rate),
                nn.Conv1d(out_channels, out_channels, kernel_size, 1, kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                SELayer1D(out_channels))

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('MaxPool', nn.MaxPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            self.shortcut.add_module('ShutConv', nn.Conv1d(in_channels, out_channels, 1, bias=False))
            self.shortcut.add_module('ShutBN', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        operation = self.operation(x)
        shortcut = self.shortcut(x)
        out = torch.relu(operation + shortcut)
        return out

class DNN(nn.Module):

    def __init__(self, num_classes=1, kernel_size=17, init_channels=12, growth_rate=16, base_channels=64,
                 stride=2, drop_out_rate=0.2):
        super(DNN, self).__init__()
        self.num_channels = init_channels

        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, kernel_size, drop_out_rate, stride)
            self.blocks.add_module("block{}_0".format(i), module)
            module = BasicBlock1D(C, C, kernel_size, drop_out_rate, 1)
            self.blocks.add_module("block{}_1".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)

        self.fc = nn.Linear(self.num_channels, num_classes)

    def forward(self, x):
        out = self.blocks(x)
        out = torch.squeeze(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":

    ecgs = torch.randn([64, 12, 7500])
    net = DNN(56)
    y = net(ecgs)
    print(y.shape)
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras/(1024 ** 2)))