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
from two import two_step_task

import torch.multiprocessing as mp


import time
device = torch.device("cuda:0")
env_name = 'GAE_Two_Step_Task_' + time.ctime(time.time())
env = two_step_task()#MAB(k)

writer = SummaryWriter("runs/"+ env_name)


input_dim = env.nb_states
hidden_dim = 48
output_dim = env.num_actions
LR = 7e-4
MAX_EP = 100000
GAMMA = 0.9

def train():
    policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr = LR)

    r = 0

    train_reward = []
    for ep in range(MAX_EP):
        ep_reward = 0

        log_prob_actions = []
        state_values = []
        rewards = []
        entropies = []
        p_action, p_reward = [0]*output_dim, 0
        
        step = 0
        a = [0]*output_dim
        rnn_state = policy.init_lstm_state()
        d = False
        t = 0
        s = env.reset()
        while not d:
            step += 1

            s = np.concatenate([s, [t]])#[a, r, step/100.0]
            s = torch.FloatTensor(s).to(device)#.unsqueeze(0)
            _policy, qvalue, rnn_state = policy(
                    s, 
                    (
                        torch.tensor(p_action).float().to(device), 
                        torch.tensor([p_reward]).float().to(device), 
                    ),
                    rnn_state
                )
            rnn_state = rnn_state[0].detach(), rnn_state[1].detach() 
            
            action_prob = F.softmax(_policy, dim=-1)
            log_prob = F.log_softmax(_policy, dim=-1)
            
            entropy = -(log_prob * action_prob).sum(1, keepdim=True)
            entropies += [entropy]
            
            dist = Categorical(action_prob)
            action = dist.sample()
            a = action.item()
            #print("Action : ", _policy)
            s1, r, d, t = env.step(a)
            p_action = np.eye(output_dim)[a]
            p_reward = r
            log_prob_actions.append(dist.log_prob(action))
            state_values.append(qvalue)
            rewards.append(r)

            ep_reward += r
            s = s1
        
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
            
            delta_t = rewards[i] + GAMMA * state_values[i + 1] - state_values[i]
            gae = gae * GAMMA * gae_lambda + delta_t
            
            action_loss = action_loss - log_prob_actions[i]*gae.detach() - 0.001 * entropies[i]
        loss = (action_loss + 0.4 * value_loss).sum()
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        if ep % 10 == 0:
            print("ep : ", ep, " | reward : ", ep_reward)

        train_reward.append(ep_reward)
        
        writer.add_scalar("Model ep reward", ep_reward, ep)
        if ep % 1000 == 0 and ep != 0:
            plot(ep)

def plot(episode_count):
        fig, ax = plt.subplots()
        x = np.arange(2)
        ax.set_ylim([0, 1.2])
        ax.set_ylabel('Stay Probability')
        
        stay_probs = env.stayProb()
        
        common = [stay_probs[0,0,0],stay_probs[1,0,0]]
        uncommon = [stay_probs[0,1,0],stay_probs[1,1,0]]
        
        ax.set_xticks([1.3,3.3])
        ax.set_xticklabels(['Last trial rewarded', 'Last trial not rewarded'])
        
        c = plt.bar([1,3],  common, color='b', width=0.5)
        uc = plt.bar([1.8,3.8], uncommon, color='r', width=0.5)
        ax.legend( (c[0], uc[0]), ('common', 'uncommon') )
        plt.savefig("results/"+env_name+ str(episode_count) + ".png")
        env.transition_count = np.zeros((2,2,2))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


train()
#plot("F")
