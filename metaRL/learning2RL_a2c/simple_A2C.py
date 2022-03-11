import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import time
import matplotlib.pyplot as plt
import numpy as np
import gym
from ActorCritic import ActorCritic
import torch.multiprocessing as mp

device = torch.device("cuda:0")
env_name = 'LunarLander-v2'
env = gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name)

input_dim = env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n
LR = 1e-3
MAX_EP = 5000
GAMMA = 0.99



def train():
#    env = gym.make(env_name)
    policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    #policy.load_state_dict(g_policy.state_dict())
    
    optimizer = optim.Adam(policy.parameters(), lr = LR)


    train_reward = []
    for ep in range(MAX_EP):
        ep_reward = 0

        log_prob_actions = []
        state_values = []
        rewards = []

        d = False
        
        s = env.reset()
        while not d:
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            state_pred, action_pred = policy(s)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()

            s, r, d, _ = env.step(action.item())
            
            log_prob_actions.append(dist.log_prob(action))
            state_values.append(state_pred)
            rewards.append(r)

            ep_reward += r
        log_prob_actions = torch.cat(log_prob_actions).to(device)
        state_values = torch.cat(state_values).squeeze(-1).to(device)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA*R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(device)
        returns = (returns - returns.mean()) / returns.std()
        
        advantage = returns - state_values
        advantage = (advantage - advantage.mean()) / advantage.std()

        advantage = advantage.detach()
        returns = returns.detach()
        action_loss = -(advantage * log_prob_actions).sum()
        value_loss = F.smooth_l1_loss(state_values, returns).sum()
        
        loss = action_loss + value_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_reward.append(ep_reward)
        
        if ep % 10 == 0: #and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-10:]), ep)

        if ep % 10 == 0:
            print("MODEL{} - EP : {} | Mean Reward : {}".format("A2C", ep, np.mean(train_reward[-10:])))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


train()

