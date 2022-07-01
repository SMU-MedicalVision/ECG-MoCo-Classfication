#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import os
import h5py
import torch
import shutil
import random
import argparse
import setproctitle

import numpy as np
import pandas as pd
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import DNN
import loss
import logs
import MSDNN
import dataset
import function
import augmentation

seed = 10000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=130)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    rootpath = os.path.dirname(os.path.abspath('..'))
    ecgpath = os.path.join(rootpath, 'ECG')
    datapath = os.path.join(rootpath, 'data')
    dsetpath = os.path.join(datapath, 'datasets')
    args.save = os.path.join(datapath, 'Classfication')
    setproctitle.setproctitle("laijiewei")

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save)

    CodeClassifyExcel = os.path.join(ecgpath, '样本标签.xls')
    codes_dict1, codes_dict2 = dataset.read_diagnostic_code_excel(CodeClassifyExcel)

    data = h5py.File(os.path.join(dsetpath, 'ECGDataSetHP.h5'), 'r')
    ecgs = data['ecgs']
    ids = np.copy(data['ids'])
    y1 = pd.DataFrame(data['y1'], columns=codes_dict1.keys())
    y2 = pd.DataFrame(data['y2'], columns=codes_dict2.keys())

    # Choose out the available diagnostics tabs in all tabs
    keys1 = ['心电图未见明显异常', '窦性心律', '房性早搏', '房性逸搏', '心房颤动', '心房扑动', '交界性逸搏', '交界性心律', '室上性心动过速',
             '室性早搏', '室性心动过速', '预激综合症', '一度房室阻滞', '二度I型房室阻滞', '束支阻滞', '二度窦房阻滞', '心腔肥厚及扩大',
             '起搏', 'ST段抬高', 'ST段压低', 'T波改变', 'T波高尖', '异常Q波', '异常q波', 'r波递增不良']
    keys2 = ['窦性心动过速', '窦性心动过缓', '显著的窦性心动过缓', '窦性心律不齐', '房性早搏未下传', '房早伴差传', '成对房性早搏',
             '加速的房性自主心律', '心房颤动伴快心室率', '心房颤动部分伴室内差异性传导', '房性心动过速', '室性早搏二联律', '心电轴左偏',
             '心电轴右偏', '不确定心电轴', '肢体导联低电压', '胸前导联低电压', '顺钟向转位', '早期复极', '左前分支阻滞', '完全性右束支阻滞',
             '不完全性右束支阻滞', '完全性左束支阻滞', '左心室高电压', '右心室高电压', 'P波增高', '心房起搏心律', '心室起搏心律',
             '房室顺序起搏心律', '心房感知心室起搏心律']

    targets1, targets2 = y1[keys1], y2[keys2]
    targets = pd.concat([targets1, targets2], axis=1)

    keys = ['心电图未见明显异常', '窦性心律', 'P波增高', '异常Q波', '异常q波', 'r波递增不良', 'ST段抬高', 'ST段压低', 'T波改变',
            'T波高尖', '窦性心动过速', '窦性心动过缓', '显著的窦性心动过缓', '窦性心律不齐', '房性早搏', '房性早搏未下传', '房早伴差传',
            '成对房性早搏', '房性逸搏', '加速的房性自主心律', '心房颤动', '心房颤动伴快心室率', '心房颤动部分伴室内差异性传导', '心房扑动',
            '交界性逸搏', '交界性心律', '室上性心动过速', '房性心动过速', '室性早搏', '室性早搏二联律', '室性心动过速', '预激综合症',
            '心电轴左偏', '心电轴右偏', '不确定心电轴', '肢体导联低电压', '胸前导联低电压', '顺钟向转位', '早期复极', '一度房室阻滞',
            '二度I型房室阻滞', '束支阻滞', '左前分支阻滞', '完全性右束支阻滞', '不完全性右束支阻滞', '完全性左束支阻滞', '二度窦房阻滞',
            '心腔肥厚及扩大', '左心室高电压', '右心室高电压',  '起搏', '心房起搏心律', '心室起搏心律', '房室顺序起搏心律', '心房感知心室起搏心律']
    targets = targets[keys]

    others = pd.DataFrame(np.zeros([len(targets), 1], dtype=np.int32), columns=['其它'])
    others[targets.sum(1) == 0] = 1
    targets = pd.concat([targets, others], axis=1)
    keys.append('其它')

    # Divide training set and test set
    nums, classes = (list(t) for t in zip(*sorted(zip(targets.sum(0), keys))))
    test_members = set()
    for i, key in enumerate(classes):
        idxs = targets.loc[targets[key] == 1, key].index.tolist()
        members = set(ids[idxs])
        num_need = len(members) // 20
        num_exis = len(test_members & members)
        members = sorted(list(members - test_members))
        test_members = test_members | set(np.random.choice(members, max(0, num_need - num_exis), replace=False))
    test_members = sorted(list(test_members))

    # Find the ECG data index by member
    train_idxs, test_idxs = list(), list()
    for i, id in enumerate(ids):
        if id in test_members:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    print('train {}, test {}'.format(len(train_idxs), len(test_idxs)))

    # Label weighting
    weights = targets.__len__() / (len(keys) * targets.sum(0).values)
    weights = torch.from_numpy(weights.astype(np.float32))

    augmentation = [transforms.RandomApply([augmentation.ECGFrequencyDropOut(rate=0.1)]),
                    transforms.RandomApply([augmentation.ECGCycleMask(rate=0.4)]),
                    transforms.RandomApply([augmentation.ECGCropResize(n=2)]),
                    transforms.RandomApply([augmentation.ECGChannelMask(masks=8)])]

    train_data = dataset.ECGDataSet(ecgs, targets, train_idxs, transform=transforms.Compose(augmentation))
    test_data = dataset.ECGDataSet(ecgs, targets, test_idxs)
    trainLoader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testLoader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = MSDNN.MSDNN(num_classes=len(keys)).cuda()
    state_dict = torch.load(os.path.join(datapath, 'ECGMoCo/moco.pth'), map_location="cpu")
    for k in list(state_dict.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        del state_dict[k]
    msc = net.load_state_dict(state_dict, False)
    print("MoCo has been loaded without {}.".format(msc))

    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))
    torch.backends.cudnn.benchmark = True
    criterion1 = torch.nn.BCEWithLogitsLoss(weight=weights)
    criterion2 = loss.LogSumExpPairwiseLoss
    lrs = function.learning_rate_seq(args.num_epochs, args.lr)
    result = logs.Logs(keys, args)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):

        optimizer.param_groups[0]['lr'] = lrs[epoch]
        train_loss = function.train(net, trainLoader, criterion1, criterion2, optimizer, epoch, args)
        test_loss, ROCAUCs, PRAUCs = function.test(net, criterion1, criterion2, testLoader, keys, args)
        result.save_logs(epoch, train_loss, test_loss, ROCAUCs, PRAUCs, net)

    result.plot_result()