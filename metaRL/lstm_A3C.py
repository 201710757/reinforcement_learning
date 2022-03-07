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
from ActorCritic_lstm import ActorCritic
import torch.multiprocessing as mp

from multi_armed_bandit import MAB

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#torch.device("cuda")
# env_name = 'CartPole-v1'
env_name = 'MultiArmedBandit'
envs = [MAB(n=3) for i in range(5001)]#gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name)

input_dim = 4#env.observation_space.shape[0]
hidden_dim = 1024
output_dim = 3#env.action_space.n
LR = 1e-3
MAX_EP = 5000

# 4 : memory error
process_num = 3

ENV_NUM = 0
def train(g_policy, model_num):
    env = envs[ENV_NUM]#gym.make(env_name)
    local_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    local_policy.load_state_dict(g_policy.state_dict())
    
    local_optimizer = optim.Adam(g_policy.parameters(), lr = LR)
    for _ in range(MAX_EP):
        train_reward = []
        
        # reset LSTM
        local_policy.reset_lstm()
        for ep in range(1):
            ep_reward = 0
            step = 0
            d = False
            a = 0
            r = 0
            env = envs[ENV_NUM]
            ENV_NUM += 1
            while not d:
                step += 1
                if step == 100:
                    d = True
                s = [0.0, a, r, d]#env.reset() 
                s = torch.FloatTensor(s).to(device).unsqueeze(0)

                action = local_policy(s)
                r = envs[ENV_NUM].pull(action.item())#env.step(action.item())
                
                gpu_reward = torch.tensor(r).type(torch.FloatTensor).to(device)
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
            
            if ep % 10 == 0 and model_num == 0:
                writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-100:]), ep)

            if ep % 100 == 0:
                print("ENV{} => MODEL{} - EP : {} | Mean Reward : {}".format(ENV_NUM, model_num, ep, np.mean(train_reward[-100:])))
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

