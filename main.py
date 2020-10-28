from Env import env
from Agent import Agent
from torch_geometric.data import Data
import torch
import numpy as np
import time

import argparse

if __name__ == "__main__":
    num_eposides = 30000
    n_step = 2
    agent = Agent(epsilon=1, eps_decay=1e-4, T=2, cuda_id=2)
    scores = [] # K是可变的，所以scores应该取相对分数
    
    t0 = time.time()
    for i in range(num_eposides):
        # 设定Env
        graph_size = 500
        # seed_size = np.random.randint(5, 100) # K的取值为 5~100
        seed_size = 50
        # p = np.random.randint(2,30)/graph_size # ER的平均度 为2~30
        p = np.log(graph_size)/graph_size
        # weight = 0.05 + np.random.rand()*0.15  # 连边权重为 0.05~0.2
        weight = 0.1
        Env = env(graph_size=graph_size, seed_size=seed_size, p=p, weight=weight) 

        mu, edge_index, edge_w, node_tag, done = Env.reset()
        graph = Data(edge_index = torch.LongTensor(edge_index),
                     mu = torch.Tensor(mu),
                     edge_w = torch.Tensor(edge_w),
                     node_tag = torch.Tensor(node_tag),
                     num_nodes = graph_size)
        graph_former_steps = []
        graph_later_steps = []
        action_steps = []
        reward_steps = []
        done_steps = []
        steps_cntr = 0

        while not done:
            graph_former_steps.append(graph)
            # choose action
            action = agent.choose_action(graph)
            action_steps.append(action)
            # env step
            _, _, _, node_tag, done, reward = Env.step(action)
            # recording
            graph.node_tag = torch.Tensor(node_tag)
            graph_later_steps.append(graph)
            reward_steps.append(reward)
            done_steps.append(done)

            steps_cntr += 1
            if steps_cntr > n_step+1:
                agent.remember(graph_former_steps[-n_step], 
                               graph_later_steps[-n_step], 
                               action_steps[-n_step],
                               sum(reward_steps[-n_step:]), 
                               done_steps[-1])
            agent.learn()
            agent.save_Q_net("Q_net.model")
            agent.save_score(scores, "score.npy")
        
        scores.append(Env.spread)

        if i > 10 and i%20==0:
            print("Eposides = {:<6}, Scores = {:.2f} Time = {:.1f}s".format(i, 
                                                    np.mean(scores[-5:]),
                                                    time.time()-t0))
            t0 = time.time()
        
