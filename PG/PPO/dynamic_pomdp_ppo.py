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

device = torch.device("cuda:1")
#env_name = 'LunarLander-v2'
env_name = 'CartPole-v1'
env = gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name + "_" + time.ctime(time.time()))

input_dim = env.observation_space.shape[0] - 1
hidden_dim = 512
output_dim = env.action_space.n
LR = 1e-4#0.0001
MAX_EP = 100000
GAMMA = 0.99
ppo_steps = 5
ppo_clip = 0.1
lmbda = 0.98

del_info = 0
def train():
    policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    policy.apply(init_weights)    
    optimizer = optim.Adam(policy.parameters(), lr = LR)

    train_reward = []
    for ep in range(MAX_EP):
        if ep % 5000 == 0:
            del_info = np.random.randint(4)
        
        ep_reward = 0

        log_prob_actions = []
        rewards = []
        states = []
        actions = []
        values = []

        d = False
        
        s = env.reset()
        while not d:
            s = np.delete(s, del_info)
            #s = np.delete(s, 2)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)

            states.append(s)

            state_pred, action_pred = policy(s)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()

            s, r, d, _ = env.step(action.item())
            
            actions.append(action)
            log_prob_actions.append(dist.log_prob(action))
            values.append(state_pred)
            rewards.append(r)

            ep_reward += r

        states = torch.cat(states).unsqueeze(0).to(device)
        actions = torch.cat(actions).unsqueeze(0).to(device)
        log_prob_actions = torch.cat(log_prob_actions).to(device)
        values = torch.cat(values).squeeze(-1).to(device)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA*R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(device)
        returns = (returns - returns.mean()) / returns.std()
        

        advantages = []
        advantage = 0
        next_value = 0
        td_err_arr = []
        for r, v in zip(reversed(rewards), reversed(values)):
            td_err = r + GAMMA * next_value - v
            td_err_arr.append(td_err.detach().cpu())
            advantage = td_err + advantage * GAMMA * lmbda
            next_value = v
            advantages.insert(0, advantage)
        advantages = torch.tensor(advantages).float().to(device)
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        
        states = states.detach()
        actions = actions.detach()
        log_prob_actions = log_prob_actions.detach()
        advantages = advantages.detach()
        returns = returns.detach()
        
        for _ in range(ppo_steps):
            s_p, a_p = policy(states)
            s_p = s_p.squeeze(-1)
            a_p = F.softmax(a_p, dim=-1)
            dist = Categorical(a_p)

            new_log_prob_actions = dist.log_prob(actions)

            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0-ppo_clip, max=1.0+ppo_clip) * advantages

            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(returns.unsqueeze(0), s_p).mean()
            
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        if ep % 10 == 0:
            #print(td_err_arr)
            writer.add_scalar("TD Err", np.mean(td_err_arr), ep)
            writer.add_scalar("advantages", advantages.mean(), ep)
            writer.add_scalar("policy loss", policy_loss.mean(), ep)
            writer.add_scalar("value loss", value_loss.mean(), ep)
            writer.add_scalar("Loss", loss.mean(), ep)
            writer.add_scalar("po - number", del_info, ep)

        train_reward.append(ep_reward)
        
        if ep % 10 == 0: #and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-10:]), ep)

        if ep % 10 == 0:
            print("MODEL{} - EP : {} | Mean Reward : {}".format(" PPO", ep, np.mean(train_reward[-10:])))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


train()

