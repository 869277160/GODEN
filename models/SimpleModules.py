import torch.nn as nn 
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 dropout=0.0):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.lins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Lin(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        for lin, batch_norm in zip(self.lins, self.batch_norms):
            lin.reset_parameters()
            batch_norm.reset_parameters()

    def forward(self, x, *args):
        for i, (lin, bn) in enumerate(zip(self.lins, self.batch_norms)):
            if i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = bn(x) if self.batch_norm else x
        return x

    def __repr__(self):
        return '{}({}, {}, num_layers={}, batch_norm={}, dropout={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_layers, self.batch_norm, self.dropout)

class SelfGating(nn.Module):
    def __init__(self, in_channels, out_channels = 1, dropout=0.0):
        super(SelfGating, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout
        self.linear = Lin(in_channels, in_channels)
        
    def forward(self, x, *args):
        return x * F.sigmoid(self.linear(x))
        
    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__, self.in_channels, self.in_channels)
