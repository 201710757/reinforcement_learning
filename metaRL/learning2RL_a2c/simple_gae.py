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


import time
device = torch.device("cuda:0")
env_name = 'GAE_Multi_Armed_Bandit_' + time.ctime(time.time())
k = 2
env = MAB(k)

writer = SummaryWriter("runs/"+ env_name)
import pandas as pd
df = pd.DataFrame()

prob_list = []
prob_list.append(env.prob)

input_dim = 3
hidden_dim = 48
output_dim = k
LR = 1e-3
MAX_EP = 10001
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
        entropies = []
        p_action, p_reward = [0]*k, 0

        step = 0
        
        #if d:
            #rnn_state = rnn_state[0].detach(), rnn_state[1].detach()
        #else:
        #rnn_state = policy.init_lstm_state()
        if ep % 100 == 0:
            rnn_state = policy.init_lstm_state()
            env = MAB(k)
            prob_list.append(env.prob)
            #print('prob : ', env.prob)
            #writer.add_hparams({'ep':int(ep/100), 'prob1':env.prob[0], 'prob2':env.prob[1]})
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
            rnn_state = rnn_state[0].detach(), rnn_state[1].detach() 
            # test version
            
            action_prob = F.softmax(action_pred, dim=-1)
            log_prob = F.log_softmax(action_pred, dim=-1)
            entropy = -(log_prob * action_prob).sum(1, keepdim=True)
            entropies += [entropy]
            try:
                #print(action_prob)
                dist = Categorical(action_prob)
            except:
                print(action_prob) 
            action = dist.sample()
            a = action.item()

            r = env.pull(a)
            p_action = np.eye(k)[a]
            p_reward = r
            log_prob_actions.append(dist.log_prob(action))
            state_values.append(state_pred)
            rewards.append(r)

            ep_reward += r
        
        #if ep % 10 == 0:
        print("prob : {} | ep {} - reward {} ".format(env.prob, ep, ep_reward))
        log_prob_actions = torch.cat(log_prob_actions).to(device)
        state_values = torch.cat(state_values).squeeze(-1).squeeze(-1).to(device)

        returns = []
        value_loss = 0
        action_loss = 0
        gae_lambda = 1.
        R = 0
        gae = torch.zeros(1, 1).to(device)

        for i in reversed(range(len(rewards)-1)):
            R = rewards[i] + GAMMA*R
            advantage = R - state_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            #print(len(rewards), state_values.shape)
            delta_t = rewards[i] + GAMMA * state_values[i + 1] - state_values[i]
            gae = gae * GAMMA * gae_lambda + delta_t
            # policy_loss = policy_loss - \
                #log_probs[i] * gae.detach() - entropy_coeff * entropies[i]
            
            action_loss = action_loss - log_prob_actions[i]*gae.detach() - 0.001 * entropies[i]
            #returns.insert(0, R)
        loss = (action_loss + 0.4 * value_loss).sum()
        #print(loss)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_reward.append(ep_reward)
        
        writer.add_scalar("Model ep reward", ep_reward, ep)

        df = pd.DataFrame(np.array(prob_list))
        df.to_csv('prob_list.csv')
        #if ep % 10 == 0:
        #    print("MODEL{} - EP : {} | Mean Reward : {}".format("A2C", ep, np.mean(train_reward[-10:])))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


train()
df = pd.DataFrame(np.array(prob_list))
df.to_csv('prob_list.csv')
