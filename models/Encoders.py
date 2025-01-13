import os, sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import config.Constants as Constants
from config.Constants import DEVICE

from models.GraphEncoder import NormalGraphNN, NormalGraphNNWithChangingEdges
from models.ODEGraphEncoder import GraphNN
from models.CoupledGraphODE import CoupledGraphODE
from models.Merger import *
from models.SimpleModules import SelfGating

sys.path.append("./")

class Encoder(nn.Module):
    def __init__(self, opt, dropout_rate=0.1, self_embedding=True):
        super(Encoder, self).__init__()
        
        #  with this interface you can try different types of graph encoders
        self.encoder = TemporalStructuralGraphEncoder(opt)

    def forward(self, inputs, input_timestamp, input_id, epoch, train=False, static_graph=None, diffusion_graph=None):
        
        user_seq_rep, all_u_embeddings = self.encoder(inputs, input_timestamp, input_id, epoch, train, static_graph, diffusion_graph)
        
        return user_seq_rep, all_u_embeddings

##################################### 基于两种图处理结构的用户编码器 #####################################
class TemporalStructuralGraphEncoder(nn.Module):
    def __init__(self, opt):
        super(TemporalStructuralGraphEncoder, self).__init__()

        # hypers
        self.ntoken = opt.token_size
        self.user_size = opt.user_size
        self.d_model = opt.d_model
        self.time_step_split = opt.time_step_split
        
        # Input node embeddings 
        self.node_embedding = nn.Embedding(self.ntoken, self.d_model, padding_idx=0)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        
        # Static graph process controls 
        self.with_st_graph = opt.with_st_graph
        self.with_normal_gnn = opt.with_normal_gnn # NOTE：for ablation study 
        
        # Static graph encoder for the model
        self.st_input_gating = SelfGating(self.d_model,self.d_model)
        self.static_graph_encoder = GraphNN(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)
        if self.with_normal_gnn: 
            self.static_graph_encoder = NormalGraphNN(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)

        # Dynamic graph user encoding control 
        self.with_dy_graph = opt.with_dy_graph 
        self.with_edge_recomputing = opt.with_edge_recomputing              # 是否重新计算边权重，即图动态性
        if self.with_edge_recomputing: assert self.with_dy_graph == True    # 如果需要动态计算图权重，则需要保证启动了动态图模块   
        self.with_dy_normal_gnn = opt.with_dy_normal_gnn                    
        self.with_dy_normal_edge_gnn = opt.with_dy_normal_edge_gnn          
        self.dy_merger = opt.dy_merger                                      
        
        # Dynamic graph user encoding module
        self.dy_input_gating = SelfGating(self.d_model,self.d_model)
        self.dy_graph_encoder = CoupledGraphODE(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)
        if not self.with_edge_recomputing:
            self.dy_graph_encoder = GraphNN(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)
        if self.with_dy_normal_gnn:
            self.dy_graph_encoder = NormalGraphNN(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)
            self.with_edge_recomputing = False
        if self.with_dy_normal_edge_gnn:
            self.dy_graph_encoder = NormalGraphNNWithChangingEdges(self.ntoken, self.d_model, dropout_rate=opt.dropout_rate, self_embedding=False)
            self.with_edge_recomputing = False

        # dynamic embedding Fusion layer
        self.dy_att = nn.Parameter(torch.ones(1, self.d_model))
        self.dy_att_m = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        init.xavier_normal_(self.dy_att.data)
        init.xavier_normal_(self.dy_att_m.data)
        
        # Out Fusion layer
        self.merger_att = nn.Parameter(torch.ones(1, self.d_model))
        self.merger_att_m = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        init.xavier_normal_(self.merger_att.data)
        init.xavier_normal_(self.merger_att_m.data)

        self.dropout = nn.Dropout(opt.dropout_rate)

    def user_st_g_rep(self, inputs, input_timestamp, input_id, epoch, train=False, static_graph=None):
        # hypers and node embeddings
        batch_size, max_len = inputs.size()
        user_embedding = self.st_input_gating(self.node_embedding.weight)
        
        # static user embedding
        static_node_embedding_lookup = self.static_graph_encoder(user_embedding, static_graph).to(DEVICE)  # [user_size, user_embedding]
        user_static_embedding = F.embedding(inputs.to(DEVICE), static_node_embedding_lookup.to(DEVICE))
        user_static_embedding = self.dropout(user_static_embedding)

        return user_static_embedding, static_node_embedding_lookup

    def user_dy_g_rep(self, inputs, input_timestamp, input_id, epoch, train, dynamic_graph=None):
        # hypers and node embeddings
        batch_size, max_len = inputs.size()
        user_embedding = self.dy_input_gating(self.node_embedding.weight)
        
        # generate user embedding
        if type(self.dy_graph_encoder) == CoupledGraphODE:
            dynamic_node_embedding_lookup, last_node_embedding_lookup = self.dy_graph_encoder(user_embedding, dynamic_graph)  # [user_size, user_embedding]
        else:
            dynamic_node_embedding_lookup = self.dy_graph_encoder(user_embedding, dynamic_graph)  # [user_size, user_embedding]

        # with edge recomputing, we can merge user states from different time steps or just use the last user state as their embedding
        if self.with_edge_recomputing:
            if self.dy_merger:
                user_dynamic_embedding, dynamic_node_embedding_lookup = self.UserDyEmbeddingMerger(inputs, dynamic_node_embedding_lookup)
            else :
                user_dynamic_embedding = F.embedding(inputs.to(DEVICE), last_node_embedding_lookup.to(DEVICE))
                user_dynamic_embedding = self.dropout(user_dynamic_embedding)
                return user_dynamic_embedding, last_node_embedding_lookup
        else: # else we replace the dynamic graph with the static graph and use the output user encoding as their embedding
            user_dynamic_embedding = F.embedding(inputs.to(DEVICE), dynamic_node_embedding_lookup.to(DEVICE))
            user_dynamic_embedding = self.dropout(user_dynamic_embedding)

        return user_dynamic_embedding, dynamic_node_embedding_lookup

    def forward(self, inputs, input_timestamp, input_id, epoch, train=False, static_graph=None, diffusion_graph=None):

        # (1) generate different types of user embeddings
        ## static user embedding
        if self.with_st_graph: 
            user_st_seq_rep, all_st_u_embeddings = self.user_st_g_rep(inputs, input_timestamp, input_id, epoch, train, static_graph)

        ## dynamic user embedding
        if self.with_dy_graph:
            user_dy_seq_rep, all_dy_u_embeddings =  self.user_dy_g_rep(inputs, input_timestamp, input_id, epoch, train, diffusion_graph)
        
        # (2) fuse the dynamic and the static embedding  
        ## 
        if self.with_dy_graph and self.with_st_graph:
            user_seq_rep, all_u_embeddings  = self.UserEmbeddingMerger(inputs, all_st_u_embeddings, all_dy_u_embeddings)
            return user_seq_rep, all_u_embeddings
        elif self.with_st_graph:
            return user_st_seq_rep, all_st_u_embeddings
        elif self.with_dy_graph: 
            return user_dy_seq_rep, all_dy_u_embeddings
        else: 
            raise "Require at least one of the encoders"
            
    def UserDyEmbeddingMerger(self, inputs, all_dy_u_embeddings):
            
        all_u_dy_embeddings, _ = self.channel_attention_1(*all_dy_u_embeddings)
        
        user_seq_rep = F.embedding(inputs.to(DEVICE), all_u_dy_embeddings.to(DEVICE))
        user_seq_rep = self.dropout(user_seq_rep)
                
        return user_seq_rep, all_u_dy_embeddings
    
    def UserEmbeddingMerger(self, inputs, all_st_u_embeddings, all_dy_u_embeddings):
        
        all_u_embeddings, _ = self.channel_attention_2(*[all_st_u_embeddings[:self.user_size], all_dy_u_embeddings[:self.user_size]])
        
        user_seq_rep = F.embedding(inputs.to(DEVICE), all_u_embeddings.to(DEVICE))
        user_seq_rep = self.dropout(user_seq_rep)
                
        return user_seq_rep, all_u_embeddings

    def channel_attention_1(self, *channel_embeddings):
    
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.dy_att, torch.matmul(embedding, self.dy_att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def channel_attention_2(self, *channel_embeddings):
    
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.merger_att, torch.matmul(embedding, self.merger_att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score







# class DualGraphEncoder(nn.Module):
#     def __init__(self, opt):
#         super(DualGraphEncoder, self).__init__()

#         # hypers
#         option = Options(opt.data_path)
#         self._ui2idx = {}
#         with open(option.ui2idx_dict, 'rb') as handle:
#             self._ui2idx = pickle.load(handle)
#         self.ntoken = len(self._ui2idx)
#         self.user_size = opt.user_size
#         self.ninp = opt.d_model
#         self.time_step_split = opt.time_step_split
#         self.with_st_graph = opt.with_st_graph
#         self.with_dy_graph = opt.with_dy_graph 
#         self.with_dy_h_merger = opt.with_dy_h_merger
#         self.with_st_merger = opt.with_st_merger
        
        
#         # input node embeddings
#         self.node_embedding = nn.Embedding(self.ntoken, self.ninp, padding_idx=0)
#         nn.init.xavier_uniform_(self.node_embedding.weight)
#         self.dropout = nn.Dropout(opt.dropout_rate)

#         # static graph encoder for the model
#         self.st_input_gating = SelfGating(self.ninp)
#         self.static_graph_encoder = GraphNN(self.ntoken, self.ninp, dropout_rate=0.15, self_embedding=False)
        
#         self.user_st_past_gru = nn.GRU(input_size = self.ninp, hidden_size = self.ninp, batch_first = True)
#         # self.past_lstm = nn.LSTM(input_size=self.ninp, hidden_size=self.ninp, batch_first=True)
#         self.user_st_short_term_att = SingleHeadTransformerBlock(input_size=self.ninp, attn_dropout=opt.dropout_rate)

#         self.user_st_attention= AttentionMerger(opt,input_size=self.ninp)
#         self.user_st_merger = Merger_ResidualWithoutFunction(opt)
        
#         self.user_st_fusion = Merger_Gate(input_size=self.ninp)


#         # dynamic graph encoder for the model
#         self.dy_input_gating = SelfGating(self.ninp)
#         self.dynamic_graph_encoder = CoupledGraphODE(self.ntoken, self.ninp, t_size = self.time_step_split, dropout_rate=0.15, self_embedding=False)
#         # self.time_attention = TimeAwareMerger(self.time_step_split, self.ninp)

#         self.dynamic_fusion_attention = nn.MultiheadAttention(embed_dim = self.ninp * self.time_step_split, num_heads = self.time_step_split * (4), batch_first=True, dropout=0.1)
#         # self.dynamic_merger = nn.Linear(self.ninp * self.time_step_split, self.ninp)
#         self.dynamic_merger = TimeAwareMerger(self.time_step_split, self.ninp)
#         self.user_dy_attention= AttentionMerger(opt,input_size=self.ninp)
#         self.user_dy_merger = Merger_ResidualWithoutFunction(opt)

#         # Out Fusion layer
#         # self.Fusion = Merger_Sigmoid(opt, input_size=self.ninp)
#         self.Fusion = Merger_Gate(input_size=self.ninp)

     
#     def user_st_rep(self, inputs, input_timestamp, input_id, epoch, train=False, static_graph=None):
        
#         # hypers and node embeddings and  user embeddings
#         batch_size, max_len = inputs.size()
#         mask = (inputs == Constants.PAD)
#         # step_len = self.time_step_split
#         node_embedding = self.node_embedding.weight
#         user_embedding = node_embedding
#         user_embedding = self.st_input_gating(node_embedding)
        
        
#         # (1) static user embedding
#         static_node_embedding_lookup = self.static_graph_encoder(
#             user_embedding, static_graph).to(DEVICE)  # [user_size, user_embedding]
#         user_static_embedding = F.embedding(
#             inputs.to(DEVICE), static_node_embedding_lookup.to(DEVICE))
#         user_static_embedding = self.dropout(user_static_embedding)
        
        
#         # 2） 将用户静态表示进行融合，忽略用户动态表示
#         if self.with_st_merger:
#             user_att_embedding, _  = self.user_st_attention(user_static_embedding, input_timestamp, input_id, train)
#             user_att_embedding = self.user_st_merger(user_static_embedding, user_att_embedding)
#             user_att_embedding = self.dropout(user_att_embedding)
#             # user_st_embedding = user_static_embedding
            
#             # 1.2） 将用户表示进行融合, 学习用户短期关联关系
#             user_cas_gru, _ = self.user_st_past_gru(user_static_embedding)
#             # user_cas_lstm, _ = self.past_lstm(input_embeddings)
#             S_cas_emb = self.user_st_short_term_att(user_cas_gru, user_cas_gru, user_cas_gru, mask=mask.to(DEVICE))
#             # S_cas_emb = self.user_st_merger(user_cas_gru, S_cas_emb)
#             S_cas_emb = self.dropout(S_cas_emb)

#             # 3) 短期用户关联和用户长期关联融合
#             user_st_embedding = self.user_st_fusion(user_att_embedding, S_cas_emb)
        
#         else:
#             user_st_embedding = user_static_embedding


#         return user_st_embedding 
              
#     def user_dy_rep(self, inputs, input_timestamp, input_id, epoch, train=False, diffusion_graph = None, dynamic_graph = None):
        
#         # hypers and node embeddings and  user embeddings
#         batch_size, max_len = inputs.size()
#         dy_con_loss = 0.0
#         step_len = self.time_step_split
#         node_embedding = self.node_embedding.weight
#         user_embedding = self.dy_input_gating(node_embedding)
        
#         dynamic_node_emb_list = self.dynamic_graph_encoder(
#             user_embedding, diffusion_graph)  # [key_size, user_size, user_embedding]
        
#         # neg_user_embedding = user_embedding[torch.randperm(user_embedding.size(0))]
#         # _, neg_dynamic_node_emb_list = self.dynamic_graph_encoder(neg_user_embedding, dynamic_graph)
        

#         # (2.2) obtain the user embeddings at each time interval for the model
#         user_dy_embedding_list = list()
#         for i, val in enumerate(sorted(dynamic_graph.keys())):
#             user_dy_embedding_sub = F.embedding(
#                 inputs.to(DEVICE), dynamic_node_emb_list[i].to(DEVICE)).unsqueeze(2)
#             user_dy_embedding_list.append(user_dy_embedding_sub)
            
            
#         user_dy_embedding = torch.cat(user_dy_embedding_list, dim=2) # [bsz, max_len, tsz, dim]
#         user_dy_embedding = self.dropout(user_dy_embedding)
#         # print(user_dy_embedding.shape)
#         # print(self.ninp * self.time_step_split)
#         # print(self.time_step_split)
#         # print(len(dynamic_node_emb_dict.keys()))
        
        
#         if self.with_dy_h_merger:
#             # (2.3)　利用多头注意力机制将各个的子图的表示进行融合
#             user_dy_embedding = user_dy_embedding.view(batch_size, max_len, -1) # [bsz, max_len, tsz, dim]
#             padding_mask = (inputs == Constants.PAD).to(DEVICE)
#             caual_mask = (torch.zeros([max_len, max_len]) == 0)
#             caual_mask = torch.triu(caual_mask, diagonal=0).to(DEVICE)
#             user_dynamic_embedding, _ = self.dynamic_fusion_attention(
#                 user_dy_embedding.to(DEVICE), user_dy_embedding.to(DEVICE), user_dy_embedding.to(DEVICE), attn_mask = caual_mask, need_weights=False, is_causal=True)
#             user_dy_embedding = user_dynamic_embedding.view(batch_size, max_len, step_len, -1) # [bsz, max_len, tsz, dim]
            
#             #  (2.4)　fuse the sub dynamic embedding to dynamic embedding
#             user_dynamic_embedding = self.dynamic_merger(user_dy_embedding, input_timestamp, dynamic_graph)
#             # user_dynamic_embedding = user_dynamic_embedding.view(batch_size, max_len, -1) # [bsz, max_len, tsz, dim]
#             # user_dynamic_embedding = self.dynamic_merger(user_dynamic_embedding)
#             # user_dynamic_embedding = self.dropout(user_dynamic_embedding)
            
#             # (2.5)　基于序列的注意力机制
#             u_dy_rep, _  = self.user_dy_attention(user_dynamic_embedding, input_timestamp, input_id, train)
#             u_dy_rep = self.user_dy_merger(user_dynamic_embedding, u_dy_rep)
#             # u_dy_rep = self.dropout(user_att_embedding)

#             return u_dy_rep
        
#         else:
#             # if not we only perform the time-aware attention for the model.
#             user_dynamic_embedding = self.dynamic_merger(user_dy_embedding, input_timestamp, dynamic_graph)

#             return u_dy_rep

        
#     def StDyMerger(self, user_static_embedding, user_dynamic_embedding):
        
    
#         user_embedding = self.Fusion(user_static_embedding, user_dynamic_embedding)
        
#         return user_embedding


#     def forward(self, inputs, input_timestamp, input_id, epoch, train=False, static_graph=None, diffusion_graph = None, dynamic_graph=None, dynamic_hyper_graph=None):

#         assert static_graph != None and dynamic_graph != None and dynamic_hyper_graph != None

#         # (1) static user embedding
#         if self.with_st_graph: 
#             u_st_rep = self.user_st_rep(inputs, input_timestamp, input_id, epoch, train, static_graph)
    
#             # (2.1) dynamic user embeddings
#         if self.with_dy_graph:
#             u_dy_rep = self.user_dy_rep(inputs, input_timestamp, input_id, epoch, train, diffusion_graph, dynamic_graph)
        
#         # print(u_dy_rep.shape)
#         # (3) fuse the dynamic and the static embedding 
#         if self.with_dy_graph and self.with_st_graph: 
#             u_rep = self.StDyMerger(u_st_rep, u_dy_rep)
            
#             return u_rep, 0 
        
#         elif self.with_st_graph and not self.with_dy_graph: 
#             return u_st_rep, 0
            
#         elif self.with_dy_graph and not self.with_st_graph: 
#             return u_dy_rep, 0
            
#         else: 
#             raise "Require at least one of the encoders"
            


if __name__ == "__main__":

        pass
 