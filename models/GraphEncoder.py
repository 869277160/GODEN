import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv

from config.Constants import DEVICE

################################## 图编码器 ##################################

######################### Normal GNN encoders #########################

class NormalGraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout_rate=0.1, self_embedding=True):
        super(NormalGraphNN, self).__init__()

        self.ntoken = ntoken
        self.input_dim = ninp
        
        self.self_embedding = self_embedding
        if self.self_embedding:
            self.embedding = nn.Embedding(self.ntoken, self.input_dim, padding_idx=0)
            init.xavier_normal_(self.embedding.weight)

        self.gnn1 = GCNConv(self.input_dim, self.input_dim * 2) 
        self.gnn2 = GCNConv(self.input_dim * 2, self.input_dim) 

        self.dropout = nn.Dropout(dropout_rate)
            
    def forward(self, node_embedding=None, graph=None):
        assert graph != None

        graph_edge_index = graph.edge_index.to(DEVICE)
        graph_x_embeddings = self.gnn1(self.embedding.weight if self.self_embedding else node_embedding, graph_edge_index) 
        graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)  # this is important for gcn to be stable
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        
        return graph_output.to(DEVICE)
    
######################### 消融实验使用的 GNN 编码模块 #########################

from models.CoupledGraphODE import EdgeEncoder, NodeEncoder

class NormalGraphNNWithChangingEdges(nn.Module):
    def __init__(self, ntoken, ninp, dropout_rate=0.1, self_embedding=True):
        super(NormalGraphNNWithChangingEdges, self).__init__()

        self.ntoken = ntoken
        self.input_dim = ninp

        self.self_embedding = self_embedding
        if self.self_embedding:
            self.embedding = nn.Embedding(self.ntoken, self.input_dim, padding_idx=0)
            init.xavier_normal_(self.embedding.weight)
        
        self.gnn1 = GCNConv(self.input_dim, self.input_dim * 2)
        
        self.gnn_node_1 = NodeEncoder(input_dim = self.input_dim*2, output_dim = self.input_dim*2, node_size = self.ntoken, dropout=0.8)
        self.gnn_edge_1 = EdgeEncoder(input_dim = self.input_dim*2, node_size = self.ntoken)
        self.gnn_node_2 = NodeEncoder(input_dim = self.input_dim*2, output_dim = self.input_dim*2, node_size = self.ntoken, dropout=0.8)
        self.gnn_edge_2 = EdgeEncoder(input_dim = self.input_dim*2, node_size = self.ntoken)
        
        self.gnn2 = GCNConv(self.input_dim * 2, self.input_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_embedding=None, graph=None):
        assert graph != None

        graph_edge_index = graph.edge_index.to(DEVICE)
        graph_x_embeddings = self.gnn1(self.embedding.weight if self.self_embedding else node_embedding, graph_edge_index) 
        graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        
        
        node_attributes = graph_x_embeddings
        edge_value = self.gnn_edge_1(node_attributes, graph_edge_index)  
        assert (not torch.isnan(edge_value).any())
        node_attributes = self.gnn_node_1(graph_x_embeddings, graph_edge_index, edge_value) # [K*N,D]

        edge_value = self.gnn_edge_2(node_attributes, graph_edge_index)  
        assert (not torch.isnan(edge_value).any())
        graph_x_embeddings = self.gnn_node_1(node_attributes, graph_edge_index, edge_value) # [K*N,D]

        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index, edge_value)
        
        return graph_output.to(DEVICE)
