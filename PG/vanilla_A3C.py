import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
import numpy as np
import gym
from ActorCritic import ActorCritic
import torch.multiprocessing as mp

device = torch.device("cuda")
env_name = 'CartPole-v1'
env = gym.make(env_name)

input_dim = env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n
LR = 1e-3
MAX_EP = 500



def train(g_policy):
    env = gym.make(env_name)
    local_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    local_policy.load_state_dict(g_policy.state_dict())
    
    local_optimizer = optim.Adam(g_policy.parameters(), lr = LR)

    train_reward = []
    for ep in range(MAX_EP):
        ep_reward = 0

        s = env.reset()
        d = False
        while not d:
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            action = local_policy(s)
            s, r, d, _ = env.step(action.item())
            
            gpu_reward = torch.tensor(r).to(device)
            local_policy.rewards.append(gpu_reward)
            ep_reward += r


        local_optimizer.zero_grad()
        loss = local_policy.loss()
        loss.backward()
        for g_param, l_param in zip(g_policy.parameters(), local_policy.parameters()):
            g_param._grad = l_param._grad
        local_optimizer.step()

        local_policy.load_state_dict(g_policy.state_dict())

        local_policy.clearMemory()
        train_reward.append(ep_reward)
        
        

        if ep % 10 == 0:
            print("EP : {} | Mean Reward : {}".format(ep, np.mean(train_reward[-10:])))
        if np.mean(train_reward[-10:]) >= 475:
            print("CLEAR!!")
            break

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


if __name__ == "__main__":
    
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    input_dim = env.observation_space.shape[0]
    hidden_dim = 1024
    output_dim = env.action_space.n

    global_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    global_policy.share_memory()

    processes = []
    process_num = 4


    global_policy.apply(init_weights)

    LR = 1e-3
    # global_optimizer = optim.Adam(global_policy.parameters(), lr = LR)
    global_policy.train()

    MAX_EP = 500

    try:
        mp.set_start_method('spawn')
        print("MP start method:",mp.get_start_method())
    except:
        pass

    for rank in range(process_num):
        p = mp.Process(target=train, args=(global_policy,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


