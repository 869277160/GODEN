import numpy as np
import torch
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from config.Constants import DEVICE

# 输入输出相同
class Merger_None(nn.Module):
    def __init__(self, opt):
        super(Merger_None, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self, a):
        return a.to(DEVICE)

# 输入输出相同
class Merger_None_2(nn.Module):
    def __init__(self, opt):
        super(Merger_None_2, self).__init__()
        # self.final_dim = opt.d_word_vec

    def forward(self, a, b):
        return a.to(DEVICE), b.to(DEVICE)

# 直接拼接之后输出
class Merger_cat(nn.Module):
    def __init__(self, opt,input_size = None):
        super(Merger_cat, self).__init__()
        
        input_size = input_size or opt.d_word_vec * 2
        self.final_dim = input_size *2
        
    def forward(self, user_dyemb, user_user_dyemb):
        dyemb = torch.cat([user_dyemb, user_user_dyemb],
                          dim=-1).to(DEVICE)  # dynamic_node_emb
        return dyemb.to(DEVICE)

# 求和方式进行筛选
class Merger_add(nn.Module):
    def __init__(self, opt,input_size = None ):
        super(Merger_add, self).__init__()
                
        input_size = input_size or opt.d_word_vec 
        self.final_dim = input_size

    def forward(self, a, b):
        dyemb = a + b
        return dyemb.to(DEVICE)

# 求差方式进行筛选
class Merger_minus(nn.Module):
    def __init__(self, opt,input_size = None ):
        super(Merger_minus, self).__init__()
        input_size = input_size or opt.d_word_vec 
        self.final_dim = input_size
        # self.final_dim = opt.d_word_vec

    def forward(self, a, b):
        dyemb = a - b
        return dyemb.to(DEVICE)

# 求点乘的方式进行筛选
class Merger_mul(nn.Module):
    def __init__(self, opt):
        super(Merger_mul, self).__init__()
        input_size = input_size or opt.d_word_vec 

    def forward(self, a, b):
        dyemb = a * b
        return dyemb.to(DEVICE)

# 一层MLP特征筛选
class Merger_MLP_Elu(nn.Module):
    def __init__(self, opt,input_size=None,dropout=0.1):
        super(Merger_MLP_Elu, self).__init__()

        input_size = input_size or opt.d_word_vec 

        self.final_dim = input_size
        
        self.linear = nn.Linear(input_size *2, self.final_dim)
        init.xavier_normal_(self.linear.weight)

        self.elu = nn.ELU()

    def forward(self, a, b):
        ui_dyemb = torch.cat([a, b], dim=-1).to(DEVICE)
        dyemb = self.elu(self.linear(ui_dyemb)).to(DEVICE)
        return dyemb.to(DEVICE)

# 一层MLP特征筛选
class Merger_MLP_Tanh(nn.Module):
    def __init__(self, opt,input_size=None, output_size= None,dropout=0.1):
        super(Merger_MLP_Tanh, self).__init__()
        
        self.input_size =  opt.d_model
        self.final_dim = opt.d_model
            
        self.linear = nn.Linear(self.input_size *2, self.final_dim)
        init.xavier_normal_(self.linear.weight)

        self.tanh = nn.Tanh()

    def forward(self, a, b):
        ui_dyemb = torch.cat([a, b], dim=-1).to(DEVICE)
        dyemb = self.tanh(self.linear(ui_dyemb)).to(DEVICE)
        return dyemb.to(DEVICE)

# 基于 sigmoid 打分进行筛选
class Merger_Sigmoid(nn.Module):
    def __init__(self, opt,input_size =None):
        super(Merger_Sigmoid, self).__init__()

        self.input_size = input_size or opt.d_word_vec 
        self.final_size = input_size or opt.d_word_vec 

        self.linear_a = nn.Linear(self.input_size, 1)
        self.linear_b = nn.Linear(self.input_size, 1)

        self.sig = nn.Sigmoid()
        
        # self.BN = nn.BatchNorm1d(se)

        init.xavier_normal_(self.linear_a.weight)
        init.xavier_normal_(self.linear_b.weight)

    def forward(self, a, b):
        score = self.sig(self.linear_a(a)+self.linear_b(b))
        # score = self.BN(score)
        
        user_embedding = score * a + (1 - score) * b
        # print(score)
        return user_embedding.to(DEVICE)
    
# 基于 sigmoid 打分进行筛选
class Merger_Gated_Sigmoid(nn.Module):
    def __init__(self, opt,input_size =None):
        super(Merger_Gated_Sigmoid, self).__init__()

        self.input_size = input_size or opt.d_word_vec 
        self.final_size = input_size or opt.d_word_vec 

        self.gated_sigmoid_linear = nn.Linear(self.input_size*2, 1)
        self.linear_output = nn.Linear(self.input_size,self.final_size)
        # self.linear_b = nn.Linear(self.input_size, 1)

        self.sig = nn.Sigmoid()

        init.xavier_normal_(self.gated_sigmoid_linear.weight)
        init.xavier_normal_(self.linear_output.weight)

    def forward(self, a, b):
        input_ab = torch.cat([a,b],dim=-1)
        # score = self.sig(self.linear_a(a)+self.linear_b(b))
        score = self.sig(self.gated_sigmoid_linear(input_ab))
        user_embedding = score * a + (1 - score) * b
        user_embedding = F.tanh(self.linear_output(user_embedding))
        
        # user_embedding = F.normalize(user_embedding, dim=-1, p=2)
        return user_embedding.to(DEVICE)
    
# DyHGCN方式进行筛选
class Merger_DyHGCN(nn.Module):
    def __init__(self, opt,input_size =None ):
        super(Merger_DyHGCN, self).__init__()

        self.input_size = input_size or opt.d_word_vec 
        self.final_size = input_size or opt.d_word_vec 

        self.linear_dyemb = nn.Linear(self.input_size*4, self.final_size)
        init.xavier_normal_(self.linear_dyemb.weight)

        self.elu = F.elu()

    def forward(self, a, b):
        # dynamic_node_emb
        dyemb = self.elu(torch.cat([a, b, a * b, a - b], dim=-1)).to(DEVICE)
        dyemb = self.elu(self.linear_dyemb(dyemb)).to(DEVICE)
        return dyemb.to(DEVICE)

#  门控机制融合 Fusion gate 
class Merger_Gate(nn.Module):
    def __init__(self, input_size = 64, output_size = 1, dropout_rate=0.2, opt= None ):
        super(Merger_Gate, self).__init__()
        
        if opt!= None:
            self.input_size = input_size or opt.d_word_vec 
        else :
            self.input_size = input_size
        
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()
        
    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
        
    def forward(self, a, b):
        emb = torch.cat([a.unsqueeze(dim = 0), b.unsqueeze(dim = 0)],dim = 0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))),dim = 0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score*emb,dim = 0)
        return out

# from the paper rethinking skip-connection 
class Merger_ResidualWithoutFunction(nn.Module):
    def __init__(self,opt,input_size = None, output_size = None, dropout_rate = 0.1):
        
        super(Merger_ResidualWithoutFunction, self).__init__()

        self.input_size = input_size or opt.d_word_vec
        self.output_size = output_size or opt.d_word_vec 

        self.LN1 = nn.LayerNorm(normalized_shape = self.input_size)
        self.LN2 = nn.LayerNorm(normalized_shape = self.input_size)
        
    
    def forward(self, input_x, input_Fx):
        
        return self.LN2(input_x + self.LN1(input_x + input_Fx)).to(DEVICE)
    

class Merger_MultiView(nn.Module):
    def __init__(self, opt,input_size = None, dropout_rate = 0.1):
        super(Merger_MultiView, self).__init__()
        self.input_size = input_size or opt.d_word_vec 
        
        self.linear1 = nn.Linear(self.input_size,self.input_size, bias=False)
        self.linear2 = nn.Linear(self.input_size,self.input_size, bias=False)
        self.linear3 = nn.Linear(self.input_size,self.input_size, bias=False)
        self.linear4 = nn.Linear(self.input_size,self.input_size, bias=False)
        # self.linear5 = nn.Linear(self.input_size,self.input_size, bias=False)

        self.linear_sigmoid1 = nn.Linear(input_size*4, 1, bias=False)
        self.linear_sigmoid2 = nn.Linear(input_size*4, 1, bias=False)
        self.linear_sigmoid3 = nn.Linear(input_size*4, 1, bias=False)         
        self.linear_sigmoid4 = nn.Linear(input_size*4, 1, bias=False)
        # self.linear_sigmoid5 = nn.Linear(input_size*4, 1, bias=False)
        
        self.linear_output = nn.Linear(input_size, input_size, bias=True)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.LN = nn.LayerNorm(normalized_shape = self.input_size)
        
    def forward(self, x1, x2, x3, x4):
        
        h1 = self.tanh(self.linear1(x1))
        h2 = self.tanh(self.linear2(x2))
        h3 = self.tanh(self.linear3(x3))
        h4 = self.tanh(self.linear4(x4))
        
        x = torch.cat((h1, h2, h3, h4), dim=-1)
        
        z1 = self.sigmoid(self.linear_sigmoid1(x))
        z2 = self.sigmoid(self.linear_sigmoid2(x))
        z3 = self.sigmoid(self.linear_sigmoid3(x))
        z4 = self.sigmoid(self.linear_sigmoid4(x))
        
        output =    z1 * x1 + \
                    z2 * x2 + \
                    z3 * x3 + \
                    z4 * x4 
        
        output = self.tanh(self.linear_output(output))
        # output = self.LN(output)
        # output = self.dropout(output)
        
        return output 
    


    def __init__(self, time_size, in_features1):
        super(TimeAwareMerger, self).__init__()
        self.time_size = time_size
        self.time_embedding = nn.Embedding(time_size, in_features1)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.1) 


    def forward(self, Dy_U_embed, input_timestamp, dynamic_node_emb_dict, mask=None, episilon=1e-6):
        '''
            T_idx: (bsz, user_len)
            Dy_U_embed: (bsz, user_len, time_len, d) # uid 从动态embedding lookup 得到的节点向量综合
            output: (bsz, user_len, d) 
        '''
        
        temperature = Dy_U_embed.size(-1) ** 0.5 + episilon
        batch_size, max_len = input_timestamp[:,:-1].size()
        step_len = self.time_size
        # input_timestamp
        
        dyemb_timestamp = torch.zeros(batch_size, max_len).long().to(DEVICE)

        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val] = i
        latest_timestamp = dynamic_node_emb_dict_time[-1]
        for t in range(0, max_len, step_len):
            try:
                la_timestamp = torch.max(
                    input_timestamp[:, t:t+step_len]).item()
                if la_timestamp < 1:
                    break
                latest_timestamp = la_timestamp
            except Exception:
                pass

            res_index = len(dynamic_node_emb_dict_time_dict)-1
            for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                if val <= latest_timestamp:
                    res_index = i
                    continue
                else:
                    break
            dyemb_timestamp[:, t:t+step_len] = res_index
        # print(dyemb_timestamp.shape)
        
        T_embed = self.time_embedding(dyemb_timestamp) # (bsz, user_len, d)
        # print(T_embed.shape)
        # print(Dy_U_embed.shape)
        affine = torch.einsum("bud,butd->but", T_embed, Dy_U_embed) # (bsz, user_len, time_len)
        score = affine / temperature 

        # if mask is None:
        #     mask = torch.triu(torch.ones(score.size()), diagonal=1).bool().to(DEVICE)
        #     score = score.masked_fill(mask, -2**32+1)

        alpha = F.softmax(score, dim=1)  # (bsz, user_len, time_len) 

        alpha = alpha.unsqueeze(dim=-1)  # (bsz, user_len, time_len, 1) 

        att = (alpha * Dy_U_embed).sum(dim=2)  # (bsz, user_len, d) 
        return att 