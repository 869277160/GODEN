import sys

sys.path.append("./")

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, reduce, repeat
from torch import einsum, nn
from torch.autograd import Variable
from torch_geometric.nn.encoding import PositionalEncoding, TemporalEncoding
import torch.nn.init as init

import config.Constants as Constants
from config.Constants import DEVICE

########################################################################## 序列位置编码 ##########################################################################

def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = Constants.PAD,
        orginal: bool = False,
):
    if orginal: 
        positional_embedding = nn.Embedding(1000, embedding_dim) # RandomStaticPositionalEmbedding for transformer
        nn.init.xavier_uniform_(positional_embedding.weight)
        return positional_embedding
    else :
        output_embedding = SinusoidalPositionalEmbedding(
                embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
            )
        return output_embedding


def make_positions(tensor, padding_idx, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at 1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).long()
    return torch.cumsum(mask, dim=1) * mask + padding_idx

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        
        self.register_buffer('_float_tensor', torch.FloatTensor(1))


    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx)
      
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1001)  # an arbitrary large number

########################################################################## 基于时间信息的序列位置编码 ##########################################################################


# Time embedding selection
class TimeEncoder(nn.Module):
    def __init__(self,opt):
        super(TimeEncoder, self).__init__()
        self.encoder_type = opt.time_encoder_type
        if opt.time_encoder_type == "Interval":
            self.time_encoder= IntervalTimeEncoder(opt)
        else:
            raise  Exception("Undefined time encoder") 

        self.output_dim = self.time_encoder.output_dim

    def forward(self, inputs, timestamp, train):
        # use interface to test more modules
        return self.time_encoder(inputs, timestamp, train)

# Time embedding with time pass encoding (current to last user)
class IntervalTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(IntervalTimeEncoder, self).__init__()

        data_name="./data/"+opt.data
        self.pass_time=opt.pass_time
        self.n_time_interval = opt.time_interval
        # if opt.data == "twitter":
        #     self.time_scale = TOSECOND
        # if opt.data == "memetracker":
        #     self.time_scale = TOWEEK
        # if opt.data == "douban":
        #     self.time_scale = TOMONTH
        # self.per_time = self.pass_time // self.n_time_interval
        self.output_dim=opt.time_dim
        self.linear_1= nn.Linear(self.n_time_interval+1, self.output_dim, bias=True).to(DEVICE)
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()


    def forward(self,inputs,timestamp,train):
        batch_size,max_len=inputs.size()

        pass_time=timestamp[:,1:]-timestamp[:,:-1]
        # pass_time = pass_time / self.time_scale
        pass_time=F.relu(((pass_time / self.pass_time) * self.n_time_interval).long())
        pass_time=pass_time.view(batch_size*max_len,1).to(DEVICE)

        # pass_time=pass_time.to(DEVICE)
        # print(pass_time.max())
        # NOTE: the process is too complicated, you can just use one-hot embedding lookup table
        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval+1).to(DEVICE)
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).to(DEVICE)
        time_embedding = self.linear_1(time_embedding_one_hot)  # [batch_size, max_len, output_dim]
        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).to(DEVICE)

        return time_embedding.to(DEVICE), timestamp[:, :-1]


if __name__ == "__main__":

    input_a = torch.randint(1,2,(16,20)) # (bsz,max_len)
    
    Pos= PositionalEmbedding(1000, 8, learned = True)
    
    output = Pos(input_a)
    
    # print(output)
    # print(output.device)
    # print(output.shape) # [bsz, max_len, pos_dim]
    
    pass 