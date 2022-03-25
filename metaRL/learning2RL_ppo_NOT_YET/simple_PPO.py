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

from two import two_step_task

device = torch.device("cuda:0")
#env_name = 'LunarLander-v2'
env_name = 'ppo_two_step_task'
env = two_step_task()#gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name + "_" + time.ctime(time.time()))

input_dim = env.nb_states#observation_space.shape[0]
hidden_dim = 48
output_dim = env.num_actions#action_space.n
LR = 1e-3#0.0001
MAX_EP = 10000
GAMMA = 0.99
ppo_steps = 5
ppo_clip = 0.1
lmbda = 0.98


def train():
    policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    policy.apply(init_weights)    
    optimizer = optim.Adam(policy.parameters(), lr = LR)

    train_reward = []
    for ep in range(MAX_EP):
        ep_reward = 0

        log_prob_actions = []
        rewards = []
        states = []
        actions = []
        values = []

        d = False
        
        s = env.reset()
        a = [0]*output_dim
        r = 0
        t = 0

        p_action, p_reward = [0]*output_dim, 0
        rnn_state = policy.init_lstm_state()
        if env.state == env.S_1:
            env.possible_switch()

        while not d:
            s = np.concatenate([s, [t]])
            s = torch.FloatTensor(s).to(device)
            states.append(s)

            state_pred, action_pred, rnn_state = policy(s, (torch.tensor(p_action).float().to(device), torch.tensor([p_reward]).float().to(device)), rnn_state)
            rnn_state = rnn_state[0].detach(), rnn_state[1].detach()

            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()

            s1, r, d, _ = env.step(action.item())
            
            p_action = np.eye(output_dim)[action.item()]
            p_reward = r
            actions.append(action)
            log_prob_actions.append(dist.log_prob(action))
            values.append(state_pred)
            rewards.append(r)

            ep_reward += r
            s = s1

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
        for r, v in zip(reversed(rewards), reversed(values)):
            td_err = r + GAMMA * next_value - v
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

        train_reward.append(ep_reward)
        
        if ep % 10 == 0: #and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-10:]), ep)

        if ep % 10 == 0:
            print("MODEL{} - EP : {} | Mean Reward : {}".format(" PPO", ep, np.mean(train_reward[-10:])))




def plot(self, episode_count):
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
        plt.savefig(plot_path +"/"+ str(episode_count) + ".png")
        env.transition_count = np.zeros((2,2,2))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


train()
plot("barPlot")
