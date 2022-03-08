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
# env_name = 'CartPole-v1'
env_name = 'LunarLander-v2'
env = gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name)

input_dim = env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n
LR = 1e-3
MAX_EP = 5000

# 4 : memory error
process_num = 3


def train(g_policy, model_num):
    env = gym.make(env_name)
    local_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    local_policy.load_state_dict(g_policy.state_dict())
    
    local_optimizer = optim.Adam(g_policy.parameters(), lr = LR)
    
#    log_prob_action = []
#    values = []
#    rewards = []
#    done = False


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
            state_pred, action_pred = local_policy(s)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            s, r, d, _ = env.step(action.item())
            #print(log_prob_action)
            log_prob_actions.append(log_prob_action)
            state_values.append(state_pred)
            rewards.append(r)

            #gpu_reward = torch.tensor(r).type(torch.FloatTensor).to(device)
            #local_policy.rewards.append(gpu_reward)
            ep_reward += r
        log_prob_actions = torch.cat(log_prob_actions).to(device)
        state_values = torch.cat(state_values).squeeze(-1).to(device)
        #print(log_prob_actions)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99*R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(device)
        returns = (returns - returns.mean()) / returns.std()
        
        #state_values = torch.tensor(state_values).to(device)
        advantage = returns - state_values
        advantage = (advantage - advantage.mean()) / advantage.std()

        #log_prob_actions = torch.tensor(log_prob_actions).to(device)
        #print(log_prob_actions)

        advantage = advantage.detach()
        returns = returns.detach()
        action_loss = -(advantage * log_prob_actions).sum()
        value_loss = F.smooth_l1_loss(state_values, returns).sum()
        

        local_optimizer.zero_grad()
        action_loss.backward(retain_graph=True)
        value_loss.backward()

        #loss = local_policy.loss()
        #loss.backward()
        for g_param, l_param in zip(g_policy.parameters(), local_policy.parameters()):
            g_param._grad = l_param._grad
        local_optimizer.step()

        local_policy.load_state_dict(g_policy.state_dict())

        #local_policy.clearMemory()
        train_reward.append(ep_reward)
        
        if ep % 10 == 0 and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-100:]), ep)

        if ep % 100 == 0:
            print("MODEL{} - EP : {} | Mean Reward : {}".format(model_num, ep, np.mean(train_reward[-100:])))
        #if np.mean(train_reward[-10:]) >= 475:
        #    print("MODEL{} - CLEAR!!".format(model_num))
        #    break

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


if __name__ == "__main__":
    

    global_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    global_policy.share_memory()

    processes = []


    global_policy.apply(init_weights)

    # global_optimizer = optim.Adam(global_policy.parameters(), lr = LR)
    global_policy.train()


    try:
        mp.set_start_method('spawn')
        print("MP start method:",mp.get_start_method())
    except:
        pass

    for rank in range(process_num):
        p = mp.Process(target=train, args=(global_policy,rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


