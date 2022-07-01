#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai


import torch

# Pair-wise Loss
def LogSumExpPairwiseLoss(y_pred, y_true):

    assert len(y_pred) == len(y_true)
    batch_size = len(y_pred)
    loss = 0

    for i in range(batch_size):
        positive = y_pred[i, y_true[i] == 1.0]
        negative = y_pred[i, y_true[i] == 0.0]

        loss_exp = torch.exp(negative.unsqueeze(1) - positive.unsqueeze(0))
        loss_sum = torch.sum(loss_exp)
        loss_log = torch.log(1 + loss_sum)

        loss += loss_log

    return 0.1 * loss / batch_size