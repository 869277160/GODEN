import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch_scatter import scatter_add
from torch_geometric.nn import Sequential, GCNConv, GATConv, GINConv

from config.Constants import DEVICE

# from torchdiffeq import odeint_adjoint as odeint
from torchdyn.core import NeuralODE
# from torchdyn.nn import DataControl, DepthCat, Augmenter
from torchdyn.utils import *

class NodeEncoder(nn.Module):
    """Node ODE function."""
    
    def __init__(self, input_dim, output_dim, node_size, dropout=0.9):
        super(NodeEncoder, self).__init__()

        self.node_size = node_size
        self.layer_norm = nn.LayerNorm(output_dim, elementwise_affine = False)
        self.dropout = nn.Dropout(dropout)
        self.gnn1 = GCNConv(input_dim, output_dim)
        
    def set_x0(self, x_0):
        self.x_0 = x_0.clone().detach()

    def forward(self, node_inputs, edge_index, edge_weights):

        '''
        :param node_inputs: [N,D] (node size, input dim)
        :param edge_index: [esz, 2], after normalize
        :param edge_weights: [esz]
        :return:
        '''
        inputs = self.layer_norm(node_inputs)
        
        x_hidden = self.gnn1(inputs, edge_index, edge_weights)
        x_hidden = self.dropout(x_hidden)

        return x_hidden - inputs

class EdgeEncoder(nn.Module):
    def __init__(self, input_dim, node_size, dropout_rate=0.1):
        super(EdgeEncoder, self).__init__()

        # hypers 
        self.input_dim = input_dim
        self.node_size = node_size
        self.dropout = nn.Dropout(p = dropout_rate)
        
        # modules 
        self.w_node2edge = nn.Linear(input_dim * 2, input_dim)  # e_ij = W([x_i||x_j])
        self.init_network_weights(self.w_node2edge)
        self.w_edge2value = nn.Sequential(
            # nn.Linear(self.input_dim * 2, self.input_dim),
            # nn.SiLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.SiLU(),
            nn.Linear(self.input_dim // 2, 1))   
        self.init_network_weights(self.w_edge2value)

    def init_network_weights(self, net, std = 0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def rel_rec_compute(self, node_inputs, edge_index):
        """map the node idx to the embeddings 

        Args:
            node_inputs (_type_): [node_size, input_dim]
            edge_index (_type_): [edge_size, 2]

        Returns:
            _type_: rel_send, rel_rec  # [edge_size, input_dim]
        """

        node_send, node_rec = edge_index[0], edge_index[1]
        
        rel_send, rel_rec = F.embedding(node_send.to(DEVICE),node_inputs.to(DEVICE)), F.embedding(node_rec.to(DEVICE),node_inputs.to(DEVICE))
        
        return rel_send, rel_rec


    def forward(self, node_inputs, edge_index):
        '''
        compute the edge representation based on the node inputs
        # NOTE: Assumes that we have the same nodes and edges across all samples.

        :param node_inputs: [node_size, input_dim] 
        :param edges: [edge_size, 2], after normalize
        :return: [edge_size,] as the new weight for edges
        '''
        d_model = node_inputs.shape[1]
        node_inputs = node_inputs.view(self.node_size, d_model)  
        
        senders, receivers = self.rel_rec_compute(node_inputs, edge_index)
        edges = torch.cat([senders, receivers], dim = -1)  # [esz,2D]

        # Compute z for edges
        edges_from_node = F.silu(self.w_node2edge(edges))  # [esz,D]
        edges_z = self.dropout(edges_from_node) # # [esz,D]

        # edge2value
        edge_2_value = torch.squeeze(F.sigmoid(self.w_edge2value(edges_z)),dim=-1) # [esz, ]

        return edge_2_value

class CoupledODEFunc(nn.Module):
    def __init__(self, node_ode_func_net, edge_ode_func_net, node_size, dropout_rate):
        """
            the ode funciton defined on the graph, which is a combination of node_ode_func_net and edge_ode_func_net
        Args:
            node_ode_func_net (_type_): the function defined on the nodes 
            edge_ode_func_net (_type_): the function defined on the edges 
            node_size (_type_): _description_
            dropout_rate (_type_): _description_
        """
        super(CoupledODEFunc, self).__init__()

        self.node_size = node_size
        self.dropout = nn.Dropout(dropout_rate)
                
        self.node_ode_func_net = node_ode_func_net  # input: node_embedding, edge_index, edge_weight
        self.edge_ode_func_net = edge_ode_func_net  # input: node_embedding, edge_index

        self.alpha = nn.Parameter(0.8 * torch.ones(self.node_size))
        self.w = nn.Parameter(torch.eye(self.edge_ode_func_net.input_dim))
        self.d = nn.Parameter(torch.ones(self.edge_ode_func_net.input_dim) )
        
    def set_edge_index(self, edge_index,):
        self.edge_index = edge_index
        
    def set_x0(self, x_0):
        self.x_0 = x_0.clone().detach()
    
    def init_network_weights(self, net, std = 0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self, input_embedding):
        """
        Perform one step in solving ODE. 
        Given current data point x and current time point t_local, 
        returns gradient dx/dt at this time point
        
        Args:
            input_embedding (_type_): the input node embeddins, in shape of [node_size, dim]

        Returns:
            _type_: _description_
        """
        
        # hyper and checkers
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1)
        assert (not torch.isnan(input_embedding).any())

        node_attributes = input_embedding
        edge_value = self.edge_ode_func_net(node_attributes, self.edge_index)  
        assert (not torch.isnan(edge_value).any())
        grad_node = self.node_ode_func_net(node_attributes, self.edge_index, edge_value) # [K*N,D]
        assert (not torch.isnan(grad_node).any())
        
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        grad_node2 = torch.einsum('il, lm->im', input_embedding, w)
        grad_node2 = self.dropout(grad_node2)

        return  alpha / 2 * grad_node - input_embedding + grad_node2 - input_embedding 


class CoupledGraphODE(nn.Module):
    def __init__(self, ntoken, input_dim, t_size = 11, dropout_rate=0.1, self_embedding=True):
        """
            # GraphODE Encoder with dynamic edge prediction

        Args:
            ntoken (_type_): _description_
            input_dim (_type_): _description_
            t_size (int, optional): _description_. Defaults to 11.
            dropout_rate (float, optional): _description_. Defaults to 0.1.
            self_embedding (bool, optional): _description_. Defaults to True.
        """
        super(CoupledGraphODE, self).__init__()
        
        # hypers for the model
        self.input_dim = input_dim
        self.token_size = ntoken
        self.t_size = t_size
        self.t_span = torch.linspace(0, 1, t_size)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.input_gnn = GCNConv(self.input_dim, self.input_dim*2)
        self.output_gnn = GCNConv(self.input_dim*2, self.input_dim)

        ## 1. Node ODE function
        self.node_ode_func_net = NodeEncoder(input_dim = self.input_dim * 2, output_dim = self.input_dim*2,node_size = self.token_size, dropout=0.8)

        ## 2. Edge ODE function
        self.edge_ode_func_net = EdgeEncoder(input_dim = input_dim * 2, node_size = self.token_size)

        ## 3. Full Function
        self.NeuralFunc =  CoupledODEFunc(
            node_ode_func_net = self.node_ode_func_net,
            edge_ode_func_net = self.edge_ode_func_net,
            node_size = self.token_size, dropout_rate=0.1)
        self.neuralDE = NeuralODE(self.NeuralFunc, solver='rk4') #
        
    
    def forward(self, node_embeddings, input_graph):
        
        # hyper and checkers
        assert input_graph != None
        edge_index = input_graph.edge_index.to(DEVICE)
        self.NeuralFunc.set_edge_index(edge_index)
        
        # (1) generate the start node state x_0
        graph_x_embeddings = self.input_gnn(node_embeddings, edge_index)
        graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)
        graph_x_embeddings = self.dropout(graph_x_embeddings)

        # (2) Neural Coupled ODE
        self.NeuralFunc.set_x0(graph_x_embeddings)
        self.node_ode_func_net.set_x0(graph_x_embeddings)
        _, graph_x_embeddings = self.neuralDE(graph_x_embeddings, t_span=self.t_span)

        # (3) Output x_t at different steps
        output_graph_embeddings = []
        for i in range(1, self.t_size):
            user_dynamic_embedding = self.output_gnn(F.silu(graph_x_embeddings[i]), edge_index).to(DEVICE)
            output_graph_embeddings.append(user_dynamic_embedding)

        return output_graph_embeddings,  output_graph_embeddings[-1]

if __name__ == '__main__':
    
    pass
