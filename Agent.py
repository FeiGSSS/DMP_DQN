"""
Created : 2020/10/26
Author  : Philip Gao
Beijing Normal University
"""
from copy import copy

from Q_s2v import Q_s2v as Q
from ReplayBuffer import replayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_max
import numpy as np
import pickle as pkl

class Agent(nn.Module):
    def __init__(self, 
                 epsilon=0.05, # 随机选择的概率
                 gamma=1, #折现因子
                 batch_size=128,  
                 lr=0.0001,
                 lr_gamma=0.999,
                 p_dim=64, 
                 T=5,
                 mem_size=1000, 
                 test=False,
                 replace_target = 50,
                 cuda_id = 5):
        super(Agent, self).__init__()        
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.p_dim = p_dim
        self.T = T
        self.mem_size = mem_size
        self.test = test

        self.pre_Q = Q(p_dim, T, lr, lr_gamma, cuda_id)
        self.target_Q = Q(p_dim, T, lr, lr_gamma, cuda_id)

        self.memory = replayBuffer(mem_size)

        self.learn_step_cntr = 0
        self.replace_target = replace_target

    def choose_action(self, graph):
        graph = copy(graph)
        graph = graph.to(self.pre_Q.device)

        # make sure select new nodes
        if np.random.rand() < self.epsilon and not self.test:
            # random selecte 
            while True:
                action = np.random.choice(graph.num_nodes, size=1)
                if graph.x[int(action)] == 0:
                    break
        else:
            Q_value = self.pre_Q(graph)
            _, q_action = torch.sort(Q_value, descending=True)
            for action in q_action:
                if graph.x[action] == 0:
                    break
        assert graph.x[int(action)] == 0
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

        # y_target = rewards + self.gamma * (1-done) * self._max_Q(graphs_later)
        y_target = rewards + self._max_Q(graphs_later)
        y_pred   = self.pre_Q(graphs_former)[self._adjust_actions(actions, graphs_former)]

        loss     = F.mse_loss(y_target, y_pred)
        loss.backward()
        
        self.pre_Q.optimizer.step()
        self.pre_Q.scheduler.step()
        self.learn_step_cntr += 1



    def _max_Q(self, graph_batch):
        Q_value = self.target_Q(graph_batch)
        return scatter_max(Q_value, graph_batch.batch)[0]

    def _adjust_actions(self, actions, batch):
        """
        adjust actions to batch
        """
        adjust_actions = []
        data_lens = [data.num_nodes for data in batch.to_data_list()]
        for i in range(self.batch_size):
            # TODO: change code for variaitional graph size
            adjust_actions.append(actions[i] + sum(data_lens[:i]))
        return adjust_actions

    def save_Q_net(self, path):
        torch.save(self.target_Q.state_dict(), path)

    def save_score(self, score, path):
        with open(path, "wb") as f:
            pkl.dump(np.array(score), f)

    def load_Q_net(self, path):
        self.target_Q.load_state_dict(torch.load(path))
        self.pre_Q.load_state_dict(torch.load(path))
