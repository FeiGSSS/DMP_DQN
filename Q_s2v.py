# -*- encoding: utf-8 -*-
'''
@File    :   Q_s2v.py
@Time    :   2020/11/08 15:31:33
@Author  :   Philip Gao 
@Contact :   feig@mail.bnu.edu.cn
@Affiliate : Beijing Normal University
'''

# here put the import lib


from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from  torch.optim.lr_scheduler import ExponentialLR

import torch_geometric as tg

from torch_scatter import scatter_add


class s2v(nn.Module):
    def __init__(self, p_dim):
        super(s2v, self).__init__()
        self.p_dim = p_dim

        self.theta1 = nn.Linear(1, p_dim, bias=False)
        self.theta2 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta3 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta4 = nn.Linear(1, p_dim, bias=False)

    def forward(self, x, mu, weight, edge_index):
        """
        x: binary scaler for each node
        mu:p_dim embedding for each node
        weight: scaler valuer for edge weight
        """
        part1 = self.theta1(x)
        part2 = self.theta2(scatter_add(mu[edge_index[0]], edge_index[1]))
        weight= F.relu(self.theta4(weight))
        part3 = self.theta3(scatter_add(weight, edge_index[1]))

        mu_update = F.relu(part1+part2+part3)

        return mu_update

class Q_s2v(nn.Module):
    def __init__(self, p_dim, T, lr, lr_gamma, cuda_id):
        super(Q_s2v, self).__init__()
        self.p_dim = p_dim
        self.T = T

        self.theta5 = nn.Linear(2*p_dim, 1, bias=False)
        self.theta6 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta7 = nn.Linear(p_dim, p_dim, bias=False)
        self.s2vs = nn.ModuleList([s2v(p_dim) for i in range(T)])

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, batch):
        x = batch.x
        weight = batch.edge_weight
        edge_index = batch.edge_index
        mu = batch.mu if hasattr(batch, "mu") else torch.zeros((edge_index.shape[1], self.p_dim)).to(self.device)

        for sub_model in self.s2vs:
            mu = sub_model(x, mu, weight, edge_index)

        pool = self.pool_repeat(x, batch)
        cat = F.relu(torch.cat((pool, self.theta7(mu)), dim=1))

        return self.theta5(cat)

    def pool(self, x, batch):
        data_lens = [data.num_nodes for data in batch.to_data_list()]
        data_pool = self.theta6(tg.nn.global_add_pool(x, batch.batch))
        pool_repeat = []
        for pool, L in zip(data_pool, data_lens):
            pool_repeat.append(pool.repeat(L, 1))
        pool_repeat = torch.cat(pool_repeat, dim=0)

        return pool_repeat
        



