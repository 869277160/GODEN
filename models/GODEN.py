import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encodings import TimeEncoder
from models.Encoders import Encoder
from models.Decoders import FusionDecoder

class GODEN(nn.Module):
    def __init__(self,opt):
        super(GODEN, self).__init__()
        
        # hypers 
        self.opt = opt
        self.dataset = opt.data
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.pos_dim = opt.pos_dim
        self.transformer_dim = opt.transformer_dim  # transformer的维度
        self.dropout_rate=opt.dropout_rate
        self.__name__ = "GODEN"
        
        # dropout module
        self.dropout = nn.Dropout(self.dropout_rate)
        self.drop_timestamp = nn.Dropout(self.dropout_rate)

        # Encoder modules 
        self.encoder = Encoder(opt)
        self.time_encoder = TimeEncoder(opt)
        self.decoder = FusionDecoder(opt)
        
        print(self)

    def forward(self, inputs, input_timestamp, input_id, epoch, train=True, static_graph=None, diffusion_graph =None):
        
        batch_size, dim = inputs.size()
        inputs = inputs[:, :-1]  # [bsz,max_len]   input_id:[batch_size]
        # mask = (inputs == Constants.PAD)
        
        # (1) graph embedding mechinaism for user representation
        user_embedding, all_token_embeddings = self.encoder(inputs, input_timestamp, input_id, epoch, train,static_graph=static_graph, diffusion_graph= diffusion_graph) # [bsz,max_len,u_dim]

        # (1.1) time embedding based on the time encoder
        # Nothing should be done here as the time is injected via one hot vector.
        time_embedding, input_timestamp = self.time_encoder(inputs, input_timestamp, train)  # [bsz,max_len,time_dim]

        # (2.1) Transformer decoder for the model
        output = self.decoder(user_embedding, time_embedding, inputs, input_timestamp, input_id, epoch, train)
        user_pred = output.view(-1, output.size(-1))  # (bsz*max_len, |U|)
        
        return user_pred  # (bsz*max_len, |U|)
