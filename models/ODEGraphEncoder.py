import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import Sequential, GCNConv, GATConv, GINConv

from torchdyn.core import NeuralODE

from config.Constants import DEVICE

class GCNLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNLayer, self).__init__()

        if input_size != output_size:
            raise AttributeError('input size must equal output size')

        # NOTE: you can try different types of conv layers 
        self.conv1 = GCNConv(input_size, input_size)
        self.conv2 = GCNConv(input_size, output_size)
        self.dropout = nn.Dropout(p=0.9) 
        self.cached_edges = torch.rand(2,100)
    
    def set_x0(self, x0):
        # NOTE: call this before forward function
        self.x0 = x0.clone().detach()    
        
    def set_edge_index(self, cached_edges):
        # NOTE: call this before forward function
        self.cached_edges = cached_edges
        
    def forward(self, x):
        x1 = self.conv1(x, self.cached_edges)
        x1 = F.normalize(x1, dim=-1, p=2)
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, self.cached_edges)
        return x2 + x # residual connection

# Static graph embedding generater
class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, t_size = 2, dropout_rate=0.1, self_embedding=False):
        super(GraphNN, self).__init__()

        self.t_size = t_size 
        self.t_span = torch.linspace(0, 1, t_size)
        self.self_embedding = self_embedding
        if self.self_embedding:
            self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
            init.xavier_normal_(self.embedding.weight)
        
        self.gnn1 = GCNConv(ninp, ninp * 2, head = 2)
        self.func = GCNLayer(input_size=ninp* 2, output_size=ninp* 2)
        self.neuralDE = NeuralODE(self.func, solver='rk4') #
        self.gnn2 = GCNConv(ninp* 2, ninp, head = 2)
        
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, node_embedding=None, graph=None):
        assert graph != None
        # if self.cached_edges == None:
        graph_edge_index = graph.edge_index.to(DEVICE)
        self.func.set_edge_index(graph_edge_index)

        graph_x_embeddings = self.gnn1(self.embedding.weight if self.self_embedding else node_embedding, graph_edge_index) 
        graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)
        graph_x_embeddings = self.dropout(graph_x_embeddings)

        self.func.set_x0(graph_x_embeddings)
        _, graph_x_embeddings = self.neuralDE(graph_x_embeddings, t_span=self.t_span)
        graph_output = self.gnn2(graph_x_embeddings[-1], graph_edge_index)

        return graph_output.to(DEVICE)
    

