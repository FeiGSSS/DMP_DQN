"""
Created : 2020/10/26
Author  : Philip Gao
Beijing Normal University
"""

# This script aims to implement the structure2vec framework used in S2V-DQN.
# Refer to Learning Combinatorial Optimization Algorithms over Graph for more 
# details.

import torch
import torch.nn as nn
import torch.optim as optim
from  torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from torch_scatter import scatter_add
from functools import partial

class S2V(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(S2V, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        Linear = partial(nn.Linear, bias=False)
        self.lin1 = Linear(1, out_dim)
        self.lin2 = Linear(1, out_dim)
        self.lin3 = Linear(in_dim, out_dim)
        self.lin4 = Linear(3*out_dim, out_dim)
        
        self.act = nn.ReLU()

    def forward(self, mu, x, edge_index, edge_w):
        r"""edge_index: directed edges"""
        """
        Message passing through the reverse direction of edges!!!
        """
        x = self.act(self.lin1(x)) # N*out_dim
        _x = x[edge_index[1, :]] # ==> E*out_dim

        edge_w = self.act(self.lin2(edge_w.unsqueeze(1))) # ==> E*out_dim
        _edge_w = edge_w # ==> E*out_dim

        mu = self.act(self.lin3(mu)) # N*out_dim
        _mu = mu[edge_index[1, :], :] # ==> E*out_dim

        cat_feat = torch.cat((_x, _edge_w, _mu), dim=1) # ==> E*3out_dim
        cat_feat_agg = scatter_add(cat_feat, edge_index[0, :], dim=0) # N*3out_dim
        cat_feat_agg = self.act(self.lin4(cat_feat_agg)) # N*out_dim

        return F.relu(x + mu + cat_feat_agg) 

# Q function
class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, lr, lr_gamma, cuda_id):
        super(Q_Fun, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.T  = T
        Linear = partial(nn.Linear, bias=False)
        self.lin5 = Linear(2*hid_dim, 1)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.lin7 = Linear(hid_dim, hid_dim)

        self.S2Vs = nn.ModuleList([S2V(in_dim=in_dim, out_dim=hid_dim)])
        for _ in range(T - 1):
            self.S2Vs.append(S2V(hid_dim, hid_dim))

        self.loss = nn.MSELoss

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, graph):
        r"""mu is node feats, x is node tag (whether selected)"""
        mu = graph.mu
        x  = graph.node_tag
        edge_w = graph.edge_w
        edge_index = graph.edge_index

        for i in range(self.T):
            mu = self.S2Vs[i](mu, x, edge_index, edge_w)
        nodes_vec = self.lin7(mu)

        if "batch" in graph.keys:
            graph_pool = scatter_add(mu, graph.batch, dim=0)[graph.batch]
        else:
            num_nodes = graph.num_nodes
            graph_pool = torch.sum(mu, dim=0, keepdim=True)
            graph_pool = graph_pool.repeat(num_nodes,1)
        
        graph_pool = self.lin6(graph_pool)
        Cat = torch.cat((graph_pool, nodes_vec), dim=1)

        return self.lin5(F.relu(Cat)).squeeze()
