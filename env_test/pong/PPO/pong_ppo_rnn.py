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
from ActorCriticRNN import ActorCritic
import torch.multiprocessing as mp

device = torch.device("cuda:0")
#env_name = 'LunarLander-v2'
env_name = 'PongDeterministic-v4'#'PongNoFrameskip-v4'#'Pong-v0'
env = gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name + "_" + time.ctime(time.time()))

input_dim = 6400 #env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n
LR = 1e-4#0.0001
MAX_EP = 1000000
GAMMA = 0.95
ppo_steps = 5
ppo_clip = 0.1
lmbda = 0.98

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

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
        hc = []
        d = False
        
        s = env.reset()
        mem_state = policy.get_init_states()
        #hc.append(mem_state)
        while not d:
            s = prepro(s)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            states.append(s)
            #s = torch.FloatTensor(s.concat())
            #print(s.shape)
            #print(mem_state.shape)
            state_pred, action_pred, mem_state = policy((s, mem_state))
            hc.append(mem_state)

            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()

            s, r, d, _ = env.step(action.item())
            
            actions.append(action)
            log_prob_actions.append(dist.log_prob(action))
            values.append(state_pred)
            rewards.append(r)

            ep_reward += r

        states = torch.cat(states).to(device)
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
        
        #mem_state = [(_hc[0].detach(), _hc[1].detach()) for _hc in hc]#torch.FloatTensor(hc).to(device)#policy.get_init_states()
        #print(np.array(mem_state).shape)
        #print(hc)

        #print(mem_state.reshape(1,-1,hidden_size))
        #print(np.reshape(mem_state,(1,-1,hidden_dim)))
        
        #mem_state = [torch.tensor((_hc[0].squeeze(1), _hc[1].squeeze(1))) for _hc in hc]
        #print(mem_state)
        
        h = hc[0][0].detach().cpu().numpy()
        c = hc[0][1].detach().cpu().numpy()
        #print(h.shape)
        for i in range(1, len(hc)):
            h = np.append(h, hc[i][0].detach().cpu().numpy(), axis=1)
            c = np.append(c, hc[i][1].detach().cpu().numpy(), axis=1)
        print(h.shape)
        mem_state = torch.FloatTensor(h, c).to(device)

        for _ in range(ppo_steps):
            s_p, a_p, mem_state = policy((states, mem_state))
            s_p = s_p.squeeze(-1)
            a_p = F.softmax(a_p, dim=-1)
            dist = Categorical(a_p)

            new_log_prob_actions = dist.log_prob(actions)

            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0-ppo_clip, max=1.0+ppo_clip) * advantages

            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(returns.unsqueeze(0), s_p.unsqueeze(0)).mean()
            
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_reward.append(ep_reward)
        
        if ep % 10 == 0: #and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-10:]), ep)
            writer.add_scalar("policy loss", policy_loss, ep)
            writer.add_scalar("value loss", value_loss, ep)
            print("MODEL{} - EP : {} | Mean Reward : {}".format(" PPO", ep, np.mean(train_reward[-10:])))
    
        if ep % 1000 == 0 and ep != 0:
            torch.save(policy.state_dict(), './ppo_pong_hugeNet_' + str(ep) + '.pth')


def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


train()

def test():
    model = ActorCritic(input_dim, hidden_dim, output_dim).to(device)

    
    model.load_state_dict(torch.load('./ppo_pong_hugeNet.pth'))
    #model = model.to(device)
    model.eval()


    MAX_EP = 10

    env = gym.make('Pong-v0')
    s = env.reset()
    tot_reward = 0
    for ep in range(MAX_EP):
        d = False
        s = env.reset()
        
        ep_reward = 0
        while not d:
            s = prepro(s)

            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            _, action_pred = model(s)

            action_prob = F.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)
            action = dist.sample()
            
            s, r, d, _ = env.step(action.item())
            env.render()
            tot_reward += r
            ep_reward += r
        print("test - ep reward : ", ep_reward)
    print("test - mean reward : ", tot_reward/MAX_EP)


test() 
