'''
Author: your name
Date: 2022-04-29 15:38:42
LastEditTime: 2024-12-01 17:36:14
LastEditors: wangding wangding19@mails.ucas.ac.cn
Description: 不同种类的损失函数
FilePath: /GODEN/utils/LossUtil.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.loss import _Loss

from config.Constants import DEVICE
import config.Constants as Constants

#  old loss compuation
def get_performance(crit, pred, gold):
    ''' Apply label smoothing 
    if needed '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    # print ("get performance, ", gold.data, pred.data)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(
        gold.ne(Constants.PAD).data).sum().float()

    true_set = set()
    for items in gold.cpu().numpy().tolist():
        true_set.add(items)
    pre_set = set()
    for item in pred.cpu().numpy().tolist():
        if item in true_set:
            pre_set.add(item)

    # if len(pre_set) / len(true_set) > 0.3 :
    #     print(gold.cpu().numpy().tolist())
    #     print(pred.cpu().numpy().tolist())

    return loss, n_correct, len(pre_set), len(true_set)


# 添加了整体损失以及评价方式计算模块
class LossComputing(_Loss):
    def __init__(self, opt):
        super(LossComputing, self).__init__()
        
        # Losses
        self.UserPrediction_loss = 0
        self.additional_loss = 0

        self.UserLoss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
        if torch.cuda.is_available():
            self.UserLoss_func = self.UserLoss_func.to(DEVICE)

    # the loss for user prediction
    def UserLoss(self, user_pred, user_gold):

        # User Prediction loss (event type)
        loss = self.UserLoss_func(user_pred, user_gold.contiguous().view(-1))
        return loss

    def forward(self, user_pred, user_gold, additional_loss=0):
        """_summary_
        Args:
            user_pred (_type_): [bsz,max_len,user_size]
            user_gold (_type_): [bsz,max_len]
            additional_loss (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
    
        self.UserPrediction_loss = self.UserLoss(user_pred, user_gold)
        self.Additional_loss = additional_loss
        
        print(f"user_loss is {self.UserPrediction_loss}, additional loss is {self.Additional_loss}.")

        loss = self.UserPrediction_loss + self.Additional_loss

        user_pred = user_pred.max(1)[1]
        user_gold = user_gold.contiguous().view(-1)
        # print ("get performance, ", gold.data, pred.data)
        n_correct = user_pred.data.eq(user_gold.data)
        n_correct = n_correct.masked_select(
            user_gold.ne(Constants.PAD).data).sum().float()

        # how many have been predicted
        true_set = set()
        for items in user_gold.cpu().numpy().tolist():
            true_set.add(items)
        pre_set = set()
        for item in user_pred.cpu().numpy().tolist():
            if item in true_set:
                pre_set.add(item)

        return loss, n_correct, len(pre_set), len(true_set)



