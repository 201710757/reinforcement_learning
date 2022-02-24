import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import MultiPro

import matplotlib.pyplot as plt
import numpy as np
import gym
device = torch.device("cuda")
class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

# env = gym.make('CartPole-v1')
worker = 8
env_name = 'CartPole-v1'
env = MultiPro.SubprocVecEnv([lambda: gym.make(env_name) for i in range(worker)])

input_dim = env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n

policy = Policy(input_dim, hidden_dim, output_dim).to(device)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
policy.apply(init_weights)

LR = 1e-3
optimizer = optim.Adam(policy.parameters(), lr = LR)

MAX_EP = 5000
GAMMA = 0.99

train_reward = []
policy.train()
for ep in range(MAX_EP):
    log_prob_actions = []
    rewards = []
    d = np.array([False])
    ep_reward = 0

    s = env.reset()

    while not d.any():
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        
        action_pred = policy(s)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = Categorical(action_prob)
        
        action = dist.sample()

        log_prob_action = dist.log_prob(action)
        # print("ACITON : ", action.reshape(-1).cpu().data.numpy())
        s, r, d, _ = env.step(action.reshape(-1).cpu().data.numpy())
        
        log_prob_actions.append(log_prob_action)
        rewards.append(r)
        
        ep_reward += r
    log_prob_actions = torch.cat(log_prob_actions)


    # REINFORCE algorithm
    normalize = True
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R*GAMMA
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    returns = returns.detach()
    loss = -(returns * log_prob_actions).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_reward.append(ep_reward)
    
    if ep % 10 == 0:
        print("EP : {} | Mean Reward : {}".format(ep, np.mean(train_reward[-10:])))
    if np.mean(train_reward[-10:]) >= 475:
        print("CLEAR!!")
        break






