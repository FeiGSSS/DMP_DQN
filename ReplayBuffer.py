import numpy as np
import torch

from torch_geometric.data import Batch

class replayBuffer():
    def __init__(self, mem_size=10000):
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.graphs_former= [None] * mem_size
        self.graphs_later = [None] * mem_size
        self.actions = [None] * mem_size
        self.rewards = [0] * mem_size
        self.done    = [None] * mem_size

    def store(self, graph_former, graph_later, action, reward, done):
        idx = self.mem_cntr % self.mem_size
        self.graphs_former[idx] = graph_former
        self.graphs_later[idx] = graph_later
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.done[idx]    = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        p = np.array(self.rewards)/np.sum(self.rewards)
        batch = np.random.choice(self.mem_size, batch_size, replace=False, p=p)
        graphs_former_batch   = Batch.from_data_list([self.graphs_former[b] for b in batch])
        graphs_later_batch = Batch.from_data_list([self.graphs_later[b] for b in batch])
        actions_batch      = torch.Tensor([self.actions[b] for b in batch])
        rewards_batch      = torch.Tensor([self.rewards[b] for b in batch])
        done_batch         = torch.Tensor([self.done[b] for b in batch])

        return graphs_former_batch, graphs_later_batch, actions_batch, rewards_batch, done_batch

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)
