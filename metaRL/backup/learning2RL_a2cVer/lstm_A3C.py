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
k = 2

env = MAB(n=k)# for i in range(5001)]#gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name)

input_dim = 3#env.observation_space.shape[0]
hidden_dim = 48
output_dim = k#env.action_space.n
LR = 1e-3
MAX_EP = 27000*3
RESET_TERM = 100*100#3000*3*3
# 4 : memory error
# 1 : A2C
process_num = 1

GAMMA = 0.8#0.99

ENV_NUM = 0
def train():
    local_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    #local_policy.load_state_dict(g_policy.state_dict())
    
    local_optimizer = optim.Adam(local_policy.parameters(), lr = LR)

    cnt = 0
    for idx in range(MAX_EP//RESET_TERM):
        train_reward = []
        ep_reward = 0
        # reset LSTM
        print("LSTM reset")
        local_policy.reset_lstm()
        for ep in range(RESET_TERM):
            ep_reward = 0
            step = 0
            d = False
            a = 0
            r = 0
            
            log_prob_actions = []
            state_values = []
            rewards = []
            actions = []

            if ep % 100 == 0:
                print("env created")
                env = MAB(n=k) 
            while not d:
                step += 1 # it's T!!
                if step == 100:
                    d = True
                if len(actions) > 0 and len(rewards) > 0 :
                    lstm_a, lstm_r = actions[-1], rewards[-1]
                else:
                    lstm_a, lstm_r = 0.0, 0.0
                s = [a, r, step, lstm_a, lstm_r]#env.reset() 
                s = torch.FloatTensor(s).to(device).unsqueeze(0)

                state_pred, action_pred = local_policy(s)
                action_prob = F.softmax(action_pred, dim=-1)
                dist = Categorical(action_prob)
                action = dist.sample()
                _action = action.item()
                a = _action

                actions.append(_action)
                r = env.pull(_action)#env.step(action.item())
                
                log_prob_actions.append(dist.log_prob(action))
                state_values.append(state_pred)
                rewards.append(r)

                ep_reward += r
            log_prob_actions = torch.cat(log_prob_actions).to(device)
            state_values = torch.cat(state_values).to(device)
            
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
            local_optimizer.zero_grad()
            loss.backward()

            #for g_param, l_param in zip(g_policy.parameters(), local_policy.parameters()):
            #    g_param._grad = l_param._grad
            local_optimizer.step()

            #local_policy.load_state_dict(g_policy.state_dict())

            train_reward.append(ep_reward)
            
            writer.add_scalar("Model - Each episode", ep_reward, cnt)
            cnt += 1
            if ep % 10 == 0:
                print("Try - {} | Env - {} => Mean Reward : {}".format(idx, ep, np.mean(train_reward)))
                train_reward = []
            
        #print(idx, " / ", MAX_EP//RESET_TERM, "   Reward : ", ep_reward)
def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


if __name__ == "__main__":
    train()
    
    import sys
    sys.exit()
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


