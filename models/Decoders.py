# basic module of transformer 
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import config.Constants as Constants
from config.Constants import DEVICE

from models.UserAdapter import AttentionMerger
from models.Merger import Merger_ResidualWithoutFunction
from models.Encodings import PositionalEmbedding

# We fuse the user aligner and the Transformer decoder into one module
class FusionDecoder(nn.Module):
    def __init__(self, opt):
        super(FusionDecoder, self).__init__()
    
        # hypers for the model
        self.input_size = opt.d_word_vec
        self.user_size = opt.user_size
        self.d_model = opt.d_word_vec
        self.transformer_dim = opt.transformer_dim  # transformer的维度
        
        # modules
        ## long term user aligner before the transformer decoder
        self.user_lt_attention= AttentionMerger(opt, input_size = self.d_model)
        self.user_lt_merger = Merger_ResidualWithoutFunction(opt)

        ## Transformer decoder 
        self.pos_embedding = PositionalEmbedding(1000,8) # use the sin pos embedding
        self.decoder = TransformerBlock(input_size = opt.transformer_dim, n_heads=opt.heads)

        self.user_predictor = nn.Linear(opt.transformer_dim, self.user_size)
        self.dropout = nn.Dropout(opt.dropout_rate)
        
    def forward(self, input_embeddings, time_embedding, inputs, input_timestamp, input_id, epoch, train=True):
        
        # hypers for the model
        batch_size, max_len, dim = input_embeddings.size()
        mask = (inputs == Constants.PAD)
        
        # 1) user aligner before the decoder 
        user_att_embedding, _  = self.user_lt_attention(input_embeddings, input_timestamp, input_id, train)
        user_att_embedding = self.user_lt_merger(input_embeddings, user_att_embedding)
        user_att_embedding = self.dropout(user_att_embedding)
        user_embedding = user_att_embedding

        # (2.1) Transformer decoder for the model
        # First compute the positional embeddings, then concate the time embedding and positional embedding with the user embedding 
        batch_t = torch.arange(inputs.size(1)).expand(inputs.size()).to(DEVICE)
        order_embed = self.dropout(self.pos_embedding(batch_t))
        user_embedding = torch.cat([user_embedding, time_embedding],dim=-1) # [bsz,max_len,u_dim_time_dim]
        final_input = torch.cat([user_embedding, order_embed], dim=-1).to(DEVICE)  # [bsz,max_len,transformer_dim] # dynamic_node_emb
        
        att_out = self.decoder(final_input.to(DEVICE), final_input.to(DEVICE), final_input.to(DEVICE), mask=mask.to(DEVICE), train = train).to(DEVICE) # [bsz,max_len,transformer_dim]
        final_out = self.dropout(att_out) # [bsz,max_len,transformer_dim]
        
        # final output of the decoder, the final layer to compute the log prob
        pred = self.user_predictor(final_out.to(DEVICE))  # (bsz, max_len, |U|)
        mask = self.get_previous_user_mask(inputs.to(DEVICE), self.user_size)
        output = pred.to(DEVICE) + mask.to(DEVICE)
        return output
        
    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.to(DEVICE)

        masked_seq = previous_mask * seqs.data.float()
        
        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.to(DEVICE)
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.to(DEVICE)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq
        
class TransformerBlock(nn.Module):
    def __init__(self, input_size, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        # self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        # print(self)

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param mask: (*, max_q_words)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5

        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)     # (batch_size, max_q_words, max_k_words)
        
        
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(DEVICE)
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        '''
        :param Q:
        :param K:
        :param V:
        :param mask: (bsz, max_q_words)
        :return:
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)


        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V, mask=None,train=False):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        # Q, K, V = x, x, x 
        
        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output

if __name__ == "__main__":
    pass 

    
    