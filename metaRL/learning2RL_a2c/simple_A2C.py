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
from multi_armed_bandit import MAB
import torch.multiprocessing as mp

device = torch.device("cuda:0")
env_name = 'Multi_Armed_Bandit'
k = 2
env = MAB(k)

writer = SummaryWriter("runs/"+ env_name)

input_dim = 3
hidden_dim = 48
output_dim = k
LR = 1e-3
MAX_EP = 40001
GAMMA = 0.99

def train():
    policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr = LR)

    d = False
    a = 0
    r = 0

    train_reward = []
    for ep in range(MAX_EP):
        ep_reward = 0

        log_prob_actions = []
        state_values = []
        rewards = []

        p_action, p_reward = [0]*k, 0

        step = 0
        
        #if d:
            #rnn_state = rnn_state[0].detach(), rnn_state[1].detach()
        #else:
        rnn_state = policy.init_lstm_state()
        d = False
        while not d:
            if step == 100:
                d = True
            step += 1

            s = [a, r, step/100.0]
            s = torch.FloatTensor(s).to(device)#.unsqueeze(0)
            state_pred, action_pred, rnn_state = policy(
                    s, 
                    (
                        torch.tensor(p_action).float().to(device), 
                        torch.tensor([p_reward]).float().to(device), 
                    ),
                    rnn_state
                )
            
            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()
            a = action.item()

            r = env.pull(a)
            p_action = np.eye(k)[a]
            p_reward = r
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

