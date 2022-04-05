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
from ActorCriticCNN import ActorCritic
import torch.multiprocessing as mp

import time

device = torch.device("cuda:0")
#env_name = 'CartPole-v1'
env_name = 'PongNoFrameskip-v4' #'Pong-v0'
env = gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name + "_" + time.ctime(time.time()))

input_dim = env.observation_space.shape[0]
hidden_dim = 512
output_dim = env.action_space.n
LR = 1e-5
MAX_EP = 1500000
GAMMA = 0.99

# 4 : memory error
process_num = 3

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I==144]=0
    I[I==109]=0
    I[I!=0]=1
    return I.astype(np.float).reshape(1,80,80)

def train(g_policy, model_num):
    env = gym.make(env_name)
    local_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    local_policy.load_state_dict(g_policy.state_dict())
    
    local_optimizer = optim.Adam(g_policy.parameters(), lr = LR)


    train_reward = []
    for ep in range(MAX_EP):
        ep_reward = 0

        log_prob_actions = []
        state_values = []
        rewards = []
        entropies = []

        d = False
        
        s = env.reset()
        while not d:
            s = prepro(s)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            state_pred, action_pred = local_policy(s)

            action_prob = F.softmax(action_pred, dim=-1)
            log_prob = F.log_softmax(action_pred, dim=-1)

            entropy = -(log_prob * action_prob).sum(1, keepdim=True)
            entropies += [entropy]

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
        value_loss = 0
        action_loss = 0
        gae_lambda = 0.99
        R = 0
        gae = torch.zeros(1,1).to(device)
        for i in reversed(range(len(rewards)-1)):
            R = rewards[i] + GAMMA*R
            advantage = R - state_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + GAMMA * state_values[i+1] - state_values[i]
            gae = gae * GAMMA * gae_lambda + delta_t

            action_loss = action_loss - log_prob_actions[i] * gae.detach() - 0.001 * entropies[i]
        
        '''
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
        '''

        loss = (action_loss + 0.4 * value_loss).sum()
        local_optimizer.zero_grad()
        loss.backward()

        for g_param, l_param in zip(g_policy.parameters(), local_policy.parameters()):
            g_param._grad = l_param._grad
        local_optimizer.step()

        local_policy.load_state_dict(g_policy.state_dict())

        train_reward.append(ep_reward)
        
        if ep % 10 == 0 and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-10:]), ep)
            writer.add_scalar("action loss", action_loss, ep)
            writer.add_scalar("value loss", value_loss, ep)


        if ep % 10 == 0:
            print("MODEL{} - EP : {} | Mean Reward : {} | Last Game Reward : {}".format(model_num, ep, np.mean(train_reward[-10:]), ep_reward))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


if __name__ == "__main__":
    
    global_policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    global_policy.share_memory()

    processes = []

    global_policy.apply(init_weights)
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


