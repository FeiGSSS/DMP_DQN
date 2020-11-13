# -*- encoding: utf-8 -*-
'''
@File    :   Env.py
@Time    :   2020/11/08 16:06:35
@Author  :   Philip Gao 
@Contact :   feig@mail.bnu.edu.cn
@Affiliate : Beijing Normal University
'''

# here put the import lib

import numpy as np
import networkx as nx

from DMP import IC
# build env for IMP

class env():
    """
    The following methods are needed:
    - reset: regenerate a instance(graph) and initialize records
    - step : take action and return state_next, reward, done 
    """
    def __init__(self, graph_size, seed_size, 
                edge_weight = 0.1, 
                random_edge_weight=False, 
                network_model="BA"):

        r"""graph_size: number of nodes
        seed_size: maximum seed set in this env
        weight: edge weight"""
        assert isinstance(graph_size, int) and graph_size > 0
        assert isinstance(seed_size, int) and seed_size > 0 and seed_size<graph_size
        self.graph_size = graph_size
        self.seed_size = seed_size
        self.edge_weight  = edge_weight
        self.random_edge_weight = random_edge_weight
        self.spread = 0
        assert network_model in {"BA", "ER"}
        self.network_model = network_model 
        self.name = "IMP"

    def reset(self):
        # 
        if self.network_model == "BA":
            self.edge_index, self.edge_weight, self.in_weight_degree, self.out_weight_degree = self._BA(self.graph_size, m=4) 
        elif self.network_model == "ER":
            exit()
        else:
            print(self.network_model, " not implemented!")

        self.simulator = IC(self.edge_index, self.edge_weight.squeeze())
        # self.simulator = simulation
        self.x         = np.zeros((self.graph_size, 1), dtype=np.float32) # scaler value for nodes states
        self.spread    = 0
        self.done      = False
        self.mu        = np.zeros((self.graph_size, 64), dtype=np.float32)   
        self.mu[:,0] = self.in_weight_degree
        self.mu[:,1] = self.out_weight_degree

        return self.edge_index, self.edge_weight, self.x, self.mu, self.done

    def step(self, action):
        action = int(action)
        assert action in range(self.graph_size)
        assert self.x[action][0] == 0
        self.x[action][0] = 1
        new_spread = self.simulator.run(self.x)
        if not  (new_spread - self.spread) > 0:
            new_spread = self.spread
        reward     = new_spread - self.spread
        self.spread = new_spread
        self.done = self._done()

        return self.x, self.done, reward

    def _BA(self, n, m):
        r"""generate undirected edge_index and edge_weight"""
        G = nx.barabasi_albert_graph(n=n, m=m)
        G = nx.classes.function.to_directed(G)
        for node in G.nodes():
            if G.in_degree(node) == 0:
                if node-1 in G.nodes:
                    G.add_node((node-1, node))
                elif node+1 in G.nodes:
                    G.add_node((node+1, node))
        edge_index = np.array(G.edges(), dtype=np.long).T
        edge_weight= self.gen_edge_weight(G.number_of_edges())

        nx.set_edge_attributes(G, {edge:w for edge, w in zip(G.edges(), edge_weight.squeeze())}, "weight")
        in_weight_degree = G.in_degree(weight="weight")
        out_weight_degree = G.out_degree(weight="weight")
        in_weight_degree = list(dict(in_weight_degree).values())
        out_weight_degree = list(dict(out_weight_degree).values())

        return edge_index, edge_weight, in_weight_degree, out_weight_degree

    def _ER(self, n, p):
        pass

    def gen_edge_weight(self, num_edges):
        r"""generate constant edge weight or random normal edge weight"""
        if not self.random_edge_weight:
            weight = np.array([self.edge_weight]*num_edges)
        else:
            weight = np.random.normal(self.edge_weight, 0.05, size=num_edges)
            weight[weight<0] = 0
        return weight.reshape(-1, 1)

    def _done(self):
        if np.sum(self.x) >= self.seed_size:
            return True
        elif (self.graph_size - self.spread)<=1:
            return True
        else:
            return False
