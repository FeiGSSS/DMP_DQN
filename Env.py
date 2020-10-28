"""
Created : 2020/10/27
Author  : Philip Gao
Beijing Normal University
"""
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
    def __init__(self, graph_size, seed_size=500, p=0.01, weight=0.05):
        # TODO:  增加env的设定参数，例如graph的模型和具体模型参数
        self.graph_size = graph_size
        self.seed_size = seed_size
        self.p = p
        self.weight = weight
        self.spread = 0
        self.name = "IMP"

    def reset(self):
        self.edge_index, self.edge_w, self.weight_degree = self._ER(self.graph_size, p=self.p, w=self.weight) 
        self.simulator = IC(self.edge_index, self.edge_w)
        self.node_tag = np.zeros((self.graph_size, 1), dtype=np.float32)
        self.spread   = self.simulator.run(self.node_tag)
        self.mu       = np.zeros((self.graph_size, 3), dtype=np.float32)
        self.mu[:, 0] = self.weight_degree
        self.done     = self._done()

        return self.mu, self.edge_index, self.edge_w, self.node_tag, self.done

    def step(self, action):
        action = int(action)
        assert action in range(self.graph_size)
        self.node_tag[action] = 1
        new_spread = self.simulator.run(self.node_tag)
        reward     = new_spread - self.spread
        self.spread = new_spread
        self.done = self._done()

        return self.mu, self.edge_index, self.edge_w, self.node_tag, self.done, reward

    def _BA(self, size, m):
        G          = nx.barabasi_albert_graph(n=size, m=m)
        # G          = G.to_directed()
        edge_index = np.array(G.edges(), dtype=np.long).T
        edge_w     = np.ones(G.number_of_edges(), dtype=np.float32) * 0.05
        # TODO: random weight
        nx.set_edge_attributes(G, {edge:w for edge, w in zip(G.edges(), edge_w)}, "weight")
        weight_degree = G.degree(weight="weight")
        weight_degree = list(dict(weight_degree).values())
        return edge_index, edge_w, weight_degree

    def _ER(self, n, p, w):
        G = nx.erdos_renyi_graph(n=n, p=p, directed=True)
        # 保证每个节点至少有一条出和入度边
        for node in G.nodes():
            if G.in_degree(node) == 0:
                G.add_edge(np.random.choice(G.nodes()), node)
            if G.out_degree(node) == 0:
                G.add_edge(node, np.random.choice(G.nodes()))

        edge_index = np.array(G.edges(), dtype=np.long).T
        edge_w     = np.ones(G.number_of_edges(), dtype=np.float32) * w
        # TODO: random weight
        nx.set_edge_attributes(G, {edge:w for edge, w in zip(G.edges(), edge_w)}, "weight")
        weight_degree = G.degree(weight="weight")
        weight_degree = list(dict(weight_degree).values())
        return edge_index, edge_w, weight_degree

    def _done(self):
        if np.sum(self.node_tag) >= self.seed_size:
            return 1
        elif (self.graph_size - self.spread)<=1:
            return 1
        else:
            return 0