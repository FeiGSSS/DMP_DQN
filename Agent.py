"""
Created : 2020/10/26
Author  : Philip Gao
Beijing Normal University
"""

from Q import Q_Fun
from ReplayBuffer import replayBuffer

import torch
import torch.nn as nn
from torch_scatter import scatter_max
import numpy as np
import pickle as pkl

class Agent(nn.Module):
    def __init__(self, 
                 epsilon=1.0, # 随机选择的概率
                 eps_decay = 1E-4, # 随机选择的概率的递减值
                 gamma=1,
                 batch_size=64, 
                 lr=0.0001,
                 lr_gamma=0.999,
                 in_dim=3, 
                 hid_dim=64, 
                 T=5,
                 mem_size=10000, 
                 test=False,
                 replace_target = 100,
                 cuda_id = 0):
        super(Agent, self).__init__()        
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.T = T
        self.mem_size = mem_size
        self.test = test

        self.pre_Q = Q_Fun(in_dim, hid_dim, T, lr, lr_gamma, cuda_id)
        self.target_Q = Q_Fun(in_dim, hid_dim, T, lr, lr_gamma, cuda_id)
        self.memory = replayBuffer(mem_size)

        self.learn_step_cntr = 0
        self.replace_target = replace_target

    def choose_action(self, graph):
        graph = graph.to(self.pre_Q.device)
        r"""Input a graph , and output a new selected node """
        graph = graph.to(self.pre_Q.device)
        Q_value = self.pre_Q(graph)

        # make sure select new nodes
        if np.random.rand() < self.epsilon and not self.test:
            # random selecte 
            self.epsilon = max(0.05, self.epsilon - self.eps_decay)
            while True:
                action = np.random.choice(graph.num_nodes, size=1)
                if graph.node_tag[int(action)] == 0:
                    break
        else:
            _, q_action = torch.sort(Q_value, descending=True)
            for action in q_action:
                if graph.node_tag[action] == 0:
                    break
        assert graph.node_tag[int(action)].item() == 0
        return int(action.item())
    
    def remember(self, *args):
        self.memory.store(*args)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # Replacing target Q net
        if self.learn_step_cntr % self.replace_target == 1:
            self.target_Q.load_state_dict(self.pre_Q.state_dict())

        graphs_former, graphs_later, actions, rewards, done = self.memory.sample(self.batch_size)
        graphs_former   = graphs_former.to(self.pre_Q.device)
        graphs_later    = graphs_later.to(self.pre_Q.device)
        actions         = actions.to(self.pre_Q.device)
        rewards         = rewards.to(self.pre_Q.device)
        done            = done.to(self.pre_Q.device)

        self.pre_Q.optimizer.zero_grad()
        
        y_target = rewards + self.gamma * (1-done) * self._max_Q(graphs_later)
        y_pred   = self.pre_Q(graphs_former)[self._idx(actions, int(graphs_former.num_nodes/self.batch_size))]
        loss     = torch.mean(torch.pow(y_target-y_pred, 2))
        loss.backward()
        self.pre_Q.optimizer.step()
        self.pre_Q.scheduler.step()
        self.learn_step_cntr += 1



    def _max_Q(self, graphs):
        batch = graphs.batch
        Q_value = self.target_Q(graphs)
        return scatter_max(Q_value, batch)[0]

    def _idx(self, actions, num_nodes):
        """
        adjust actions to batch
        """
        adjust_actions = []
        for i in range(self.batch_size):
            # TODO: change code for variaitional graph size
            adjust_actions.append(actions[i] + num_nodes*i)
        return adjust_actions

    def save_Q_net(self, path):
        torch.save(self.target_Q.state_dict(), path)

    def save_score(self, score, path):
        with open(path, "wb") as f:
            pkl.dump(np.array(score), f)

    def load_Q_net(self, path):
        self.target_Q.load_state_dict(torch.load(path))
        self.pre_Q.load_state_dict(torch.load(path))
