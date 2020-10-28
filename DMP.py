import networkx as nx
import torch as T
from torch_scatter import scatter_mul, scatter_add
import pickle as pkl
import time

class IC():
    def __init__(self, edge_list, edge_w, device="cpu", max_iter=10): 
        self.device = device

        self.src_nodes = T.LongTensor(edge_list[0]).to(device)
        self.tar_nodes = T.LongTensor(edge_list[1]).to(device)
        self.weights   = T.FloatTensor(edge_w).to(device)
        self.cave_index = T.LongTensor(self.cave(edge_list)).to(device)
        
        self.N = max([T.max(self.src_nodes), T.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)
        self.out_weight_d = scatter_add(self.weights, self.src_nodes).to(device)

        self.max_iter = max_iter

    def cave(self, edge_list):
        G = nx.DiGraph()
        edge_list = [(s,t) for s, t in zip(*edge_list)]
        G.add_edges_from(edge_list)
        attr = {edge:w for edge, w in zip(edge_list, range(len(edge_list)))}
        nx.set_edge_attributes(G, attr, "idx")

        cave = []
        for edge in edge_list:
            if G.has_edge(*edge[::-1]):
                cave.append(G.edges[edge[::-1]]["idx"])
            else:
                cave.append(len(edge_list))
        return cave

    def _set_seeds(self, seed_list):
        self.seeds = seed_list if T.is_tensor(seed_list) else T.Tensor(seed_list)
        self.seeds = self.seeds.to(self.device)
        self.Ps_i_0 = 1 - self.seeds

        self.Theta_0 = T.ones(self.E).to(self.device)        # init Theta(t=0)
        self.Ps_0 = 1 - self.seeds[self.src_nodes]    # Ps(t=0)
        self.Phi_0 = 1 - self.Ps_0 # init Thetau(t=0)

        self.Theta_t = self.Theta_0 - self.weights * self.Phi_0 + 1E-10 #get rid of NaN
        self.Ps_t_1 = self.Ps_0             # Ps(t-1)
        self.Ps_t = self.Ps_0 * self.mulmul(self.Theta_t) # Ps(t)
        self.inf_log = [self.seeds.sum(), self.influence()]


    def mulmul(self, Theta_t):
        Theta = scatter_mul(Theta_t, index=self.tar_nodes) # [N]
        Theta = Theta[self.src_nodes] #[E]
        Theta_cav = scatter_mul(Theta_t, index=self.cave_index)[:self.E]

        mul = Theta / Theta_cav
        return mul

    def forward(self):
        Phi_t = self.Ps_t_1 - self.Ps_t
        self.Theta_t = self.Theta_t - self.weights * Phi_t
        Ps_new = self.Ps_0 * self.mulmul(self.Theta_t)

        self.Ps_t_1 = self.Ps_t
        self.Ps_t   = Ps_new
    
    def influence(self):
        # Ps_i : the probability of node i being S 
        self.Ps_i = self.Ps_i_0 * scatter_mul(self.Theta_t, index=self.tar_nodes)
        return T.sum(1-self.Ps_i)
        
    def theta_aggr(self):
        theta = scatter_mul(self.Theta_t, index=self.tar_nodes)

        return theta, self.Ps_i

    def run(self, seed_list):
        seed_list = seed_list.squeeze()
        if len(seed_list) != self.N:
            _seed_list = T.zeros(self.N)
            _seed_list[seed_list] = 1
            seed_list = _seed_list
        self._set_seeds(seed_list)
        for _ in range(self.max_iter):
            self.forward()
            new_inf = self.influence()

            if abs(new_inf - self.inf_log[-1]) < 1.0:
                break
            else:
                self.inf_log.append(new_inf)

        return self.inf_log[-1].numpy()

