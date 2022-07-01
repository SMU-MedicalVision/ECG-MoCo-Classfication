#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Save training and testing logs
class Logs(object):

    def __init__(self, keys, args):
        self.keys = keys
        self.args = args
        self.columns1 = ['train_loss', 'test_loss', 'epoch', 'best_test_loss']
        self.columns2 = ['ROCAUC_' + key for key in self.keys]
        self.columns3 = ['PRAUC_' + key for key in self.keys]
        self.result = pd.DataFrame(columns=self.columns1 + self.columns2 + self.columns3)
        self.result.loc[0, 'best_test_loss'] = None

    # Save the model with the lowest test set loss
    def save_logs(self, epoch, train_loss, test_loss, ROCAUCs, PRAUCs, net):
        self.result.loc[epoch, 'train_loss'] = train_loss
        self.result.loc[epoch, 'test_loss'] = test_loss

        for i, key in enumerate(self.keys):
            self.result.loc[epoch, 'ROCAUC_' + key] = ROCAUCs[i]
            self.result.loc[epoch, 'PRAUC_' + key] = PRAUCs[i]

        if self.result.loc[0, 'best_test_loss'] == None:
            print("The test loss of the model is {:.4f}.\n".format(test_loss))
            self.result.loc[0, 'epoch'] = epoch
            self.result.loc[0, 'best_test_loss'] = test_loss
            torch.save(net.state_dict(), os.path.join(self.args.save, 'model.pth'))
        elif self.result.loc[0, 'best_test_loss'] > test_loss:
            print("The test loss of the model is improved from {:.4f} to {:.4f}.\n".format(
                self.result.loc[0, 'best_test_loss'], test_loss))
            self.result.loc[0, 'epoch'] = epoch
            self.result.loc[0, 'best_test_loss'] = test_loss
            torch.save(net.state_dict(), os.path.join(self.args.save, 'model.pth'))
        else:
            print("The test loss of the model didn't improved from {:.4f}.\n".format(self.result.loc[0, 'best_test_loss']))

        self.result.to_excel(os.path.join(self.args.save, 'result.xlsx'))

    def plot_result(self, name='result'):

        epoch = self.result.loc[0, 'epoch']

        plt.figure()
        plt.semilogy()
        plt.plot(self.result['train_loss'], label='train_loss')
        plt.plot(self.result['test_loss'], label='test_loss')
        plt.scatter(epoch, self.result.loc[epoch, 'train_loss'], linewidth=2)
        plt.scatter(epoch, self.result.loc[epoch, 'test_loss'], linewidth=2)
        plt.legend(loc=2)

        plt.grid()
        plt.title('Best test loss: {:.4f}'.format(self.result.loc[0, 'best_test_loss']))

        plt.savefig(os.path.join(self.args.save, name + '1'), dpi=500)
        plt.clf()
        plt.close()

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure()
        plt.grid()
        roc_auc_mean = list()
        for key in self.keys:
            roc_auc = self.result.loc[epoch, 'ROCAUC_' + key]
            plt.plot(self.result['ROCAUC_' + key], label="{:.4f} ".format(roc_auc) + key)
            roc_auc_mean.append(roc_auc)
        plt.legend(bbox_to_anchor=(1.5, 1.5), ncol=2)
        roc_auc_mean.pop()
        roc_auc_mean = np.mean(roc_auc_mean)
        plt.title('Best ROCAUC: {:.4f}'.format(roc_auc_mean))

        plt.savefig(os.path.join(self.args.save, name + '2'), dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure()
        plt.grid()
        pr_auc_mean = list()
        for key in self.keys:
            pr_auc = self.result.loc[epoch, 'PRAUC_' + key]
            plt.plot(self.result['PRAUC_' + key], label="{:.4f} ".format(pr_auc) + key)
            pr_auc_mean.append(pr_auc)
        plt.legend(bbox_to_anchor=(1.5, 1.5), ncol=2)
        pr_auc_mean.pop()
        pr_auc_mean = np.mean(pr_auc_mean)
        plt.title('Best PRAUC: {:.4f}'.format(pr_auc_mean))

        plt.savefig(os.path.join(self.args.save, name + '3'), dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close()