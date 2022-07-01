#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import sys
import torch

import numpy as np
import sklearn.metrics as M

from torch.autograd import Variable

# The train function of the classification network
def train(net, trainLoader, criterion1, criterion2, optimizer, epoch, args):

    net.train()
    nProcessed = 0
    train_loss = list()
    nTrain = len(trainLoader.dataset)
    criterion1.cuda()

    for batch_idx, (batch_data, batch_label) in enumerate(trainLoader):

        batch_data = Variable(batch_data).cuda()
        batch_label = Variable(batch_label).cuda()

        optimizer.zero_grad()
        batch_logits = net(batch_data)
        batch_loss = criterion1(batch_logits, batch_label) + criterion2(batch_logits, batch_label)
        batch_loss.backward()
        optimizer.step()

        nProcessed += len(batch_data)
        partialEpoch = epoch + batch_idx / len(trainLoader)
        train_loss.append(batch_loss.item())

        sys.stdout.write('\r')
        sys.stdout.write('Train Epoch: {:.2f}|{} [{}/{}] \t Loss: {:.4f}'.format(
            partialEpoch, args.num_epochs, nProcessed, nTrain, batch_loss.item()))
        sys.stdout.flush()

    train_loss = np.mean(train_loss)
    sys.stdout.write('\r')
    sys.stdout.write('Train Epoch: {:.2f}|{} [{}/{}] \t Loss: {:.4f}'.format(
        epoch + 1, args.num_epochs, nProcessed, nTrain, train_loss))
    sys.stdout.flush()

    return train_loss

# Test test function for classification network
def test(net, criterion1, criterion2, testLoader, keys, args):

    net.eval()
    criterion1.cpu()

    labels = torch.empty([len(testLoader.dataset), len(keys)])
    logits = torch.empty([len(testLoader.dataset), len(keys)])

    with torch.no_grad():

        for batch_idx, (batch_data, batch_label) in enumerate(testLoader):

            batch_size = len(batch_data)
            batch_data = Variable(batch_data).cuda()
            logits[batch_idx*args.batch_size:batch_idx*args.batch_size+batch_size] = net(batch_data).data.cpu()
            labels[batch_idx*args.batch_size:batch_idx*args.batch_size+batch_size] = batch_label

    test_loss = criterion1(logits, labels).item() + criterion2(logits, labels).item()
    ROCAUCs, PRAUCs = list(), list()
    for i, key in enumerate(keys):
        ROCAUCs.append(M.roc_auc_score(labels[:, i].data.numpy(), logits[:, i].data.numpy()))
        PRAUCs.append(M.average_precision_score(labels[:, i].data.numpy(), logits[:, i].data.numpy()))

    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    return test_loss, ROCAUCs, PRAUCs

# one-cycle learning rate
def learning_rate_seq(num_epochs=130, learning_rate=0.01):

    one_cycle = num_epochs - 30
    half_len = int(one_cycle * 0.45)
    x1 = np.linspace(0.1 * learning_rate, learning_rate, half_len)
    x2 = np.linspace(x1[-1], 0.1 * learning_rate, half_len + 1)[1:]
    x3 = np.linspace(x2[-1], 0.001 * learning_rate, one_cycle - 2 * half_len + 1)[1:]

    x4 = np.linspace(0.5 * learning_rate, 0.001 * learning_rate, 10)

    x = np.concatenate([x1, x2, x3, x4, x4, x4])

    return x

def get_labels_logits(net, testLoader, keys, args):

    net.eval()

    labels = torch.empty([len(testLoader.dataset), len(keys)])
    logits = torch.empty([len(testLoader.dataset), len(keys)])

    with torch.no_grad():

        for batch_idx, (batch_data, batch_label) in enumerate(testLoader):
            batch_size = len(batch_data)
            batch_data = Variable(batch_data).cuda()
            logits[batch_idx * args.batch_size:batch_idx * args.batch_size + batch_size] = net(batch_data).data.cpu()
            labels[batch_idx * args.batch_size:batch_idx * args.batch_size + batch_size] = batch_label

    return labels, logits