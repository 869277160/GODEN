import numpy as np
import torch
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from config.Constants import DEVICE


class NoAdapter(nn.Module):
    def __init__(self, opt, input_size= None):
        super(NoAdapter, self).__init__()
        self.input_size = input_size or opt.d_word_vec 
        self.output_size = input_size or opt.d_word_vec 
    
    def forward(self,user_embedding, timestamp, input_id, train,time_embedding,mask):
         
        return user_embedding, 0

# 级联内自注意力机制 (with mask)
class AttentionMerger(nn.Module):
    def __init__(self, opt, input_size = 64,):
        super(AttentionMerger, self).__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.linear_1 = nn.Linear(self.input_size, self.output_size)
        init.xavier_normal_(self.linear_1.weight)
        self.linear_2 = nn.Linear(self.input_size, self.output_size)
        init.xavier_normal_(self.linear_2.weight)

    def forward(self, user_embedding, timestamp, input_id, train):
        
        batch_size, max_len, dim = user_embedding.size()
        Q = F.tanh(self.linear_1(user_embedding))
        K = F.tanh(self.linear_2(user_embedding))
        Q_K = torch.einsum("bld,bmd->bml", Q, K).to(DEVICE)
            
        # self attention 
        temperature = dim ** 0.5
        episilon = 1e-6
        Q_K = Q_K / (temperature + episilon)
        mask = torch.zeros([max_len, max_len]).to(DEVICE)
        mask += -2**32+1
        mask = torch.triu(mask, diagonal=1).to(DEVICE)

        b_mask = torch.zeros_like(Q_K).to(DEVICE)
        b_mask[:, :, :] = mask[:, :]

        Q_K += b_mask
        score = F.softmax(Q_K, dim=-1).to(DEVICE)
        output = torch.einsum("bll,bmd->bld", score, user_embedding)

        return output, 0 


