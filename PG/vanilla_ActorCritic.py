import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
import numpy as np
import gym
from ActorCritic import ActorCritic

device = torch.device("cuda")

env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n

policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
policy.apply(init_weights)

LR = 1e-3
optimizer = optim.Adam(policy.parameters(), lr = LR)
policy.train()

MAX_EP = 500
# GAMMA = 0.99

train_reward = []
for ep in range(MAX_EP):
    ep_reward = 0

    s = env.reset()
    d = False
    while not d:
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        action = policy(s)
        s, r, d, _ = env.step(action)
        
        # gpu_reward = torch.FloatTensor(r).to(device)
        policy.rewards.append(r)
        ep_reward += r


    optimizer.zero_grad()
    loss = policy.loss()
    loss.backward()
    optimizer.step()

    policy.clearMemory()

    train_reward.append(ep_reward)
    
    if ep % 10 == 0:
        print("EP : {} | Mean Reward : {}".format(ep, np.mean(train_reward[-10:])))
    if np.mean(train_reward[-10:]) >= 475:
        print("CLEAR!!")
        break






