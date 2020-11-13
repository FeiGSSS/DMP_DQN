from Env import env
from Agent import Agent
import torch_geometric as tg
from torch_geometric.data import Data
import torch
import numpy as np
import time
from copy import copy

import argparse

if __name__ == "__main__":
    num_eposides = 300000
    n_step = 1
    t0 = time.time()

    
    agent = Agent(cuda_id=3,
                  epsilon=1, # 随机选择的初始概率
                  epsilon_decay=1E-5, # 随机选择的概率decay
                  batch_size=256,
                  lr=0.01,
                  T=3,
                  mem_size=50000)
    scores = [] 
    
    
    for i in range(num_eposides):
        # 设定Env
        # graph_size = np.random.randint(50, 100)
        # seed_size = np.random.randint(10, 30)
        graph_size = 100
        seed_size = 25
        Env = env(graph_size=graph_size, seed_size=seed_size, edge_weight=0.1,
                  random_edge_weight=False, network_model="BA")
        edge_index, edge_weight, x, mu, done = Env.reset()
        graph = Data(edge_index = torch.LongTensor(edge_index),
                     edge_weight = torch.Tensor(edge_weight),
                     mu = torch.Tensor(mu),
                     x = torch.Tensor(x))
        # to be stored
        graph_former_steps = []
        graph_later_steps = []
        action_steps = []
        reward_steps = []
        done_steps = []
        steps_cntr = 0

        # running eposide
        while not done:
            graph_former_steps.append(copy(graph))
            # choose action
            action = agent.choose_action(graph)
            action_steps.append(copy(action))
            # env step
            x, done, reward = Env.step(action)
            # recording
            reward_steps.append(copy(reward))
            graph.x = torch.Tensor(x)
            graph_later_steps.append(copy(graph))
            done_steps.append(copy(done))

            steps_cntr += 1
            if steps_cntr >= n_step:
                agent.remember(graph_former_steps[-n_step], 
                               graph_later_steps[-1], 
                               action_steps[-n_step],
                               sum(reward_steps[-n_step:]), 
                               done_steps[-1])
            agent.learn()
        scores.append(Env.spread)
        agent.save_Q_net("checkpoints/Q_net.model")
        agent.save_score(scores, "results/score.npy")

        if i >10 and i % 10 == 0:
            print("Eposides = {:<6}, Scores = {:.2f} Time = {:.1f}s".format(i, np.mean(scores[-1]),time.time()-t0))
            t0 = time.time()
