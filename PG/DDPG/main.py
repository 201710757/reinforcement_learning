import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import gym

from Network import Q, Mu
from OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise
from ReplayBuffer import ReplayBuffer


device = torch.device("cuda:0")
LR_Q = 0.001
LR_Mu = 0.0005
GAMMA = 0.99
tau = 0.005
MAX_STEPS = 10000
batch_size = 32

def train(mu, mu_target, q, q_target, memory, q_optim, mu_optim):
    s, a, r, sp, d = memory.sample(batch_size)

    target = r + GAMMA * q_target(sp, mu_target(sp)) * d
    critic_loss = F.mse_loss(target.detach(), q(s, a))
    q_optim.zero_grad()
    critic_loss.backward()
    q_optim.step()

    actor_loss = -q(s, mu(s)).mean()
    mu_optim.zero_grad()
    actor_loss.backward()
    mu_optim.step()

def soft_update(net, net_grad):
    for p_t, p in zip(net_target.parameters(), net.parameters()):
        p_t.data.copy_(tau * p.data + (1.0 - tau) * p_t.data)

def main():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    input_dim = env.observation_space.shape[0]

    memory = ReplayBuffer()

    q, q_target = Q(input_dim).to(device), Q(input_dim).to(device)
    mu, mu_target = Mu(input_dim).to(device), Mu(input_dim).to(device)

    q_target.load_state_dict(q.state_dict())
    mu_target.load_state_dict(mu.state_dict())

    q_optim = optim.Adam(q.parameters(), lr = LR_Q)
    mu_optim = optim.Adam(mu.parameters(), lr = LR_Mu)
    N = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    total_score = 0
    for ep in range(MAX_STEPS):
        s = env.reset()
        d = False

        while not d:
            a = mu(torch.FloatTensor(s).to(device))
            noise = N()[0]
            print(a, noise)
            a = a.item() + noise#N()[0]
            
            sp, r, d, _ = env.step([a])
            memory.put((s, a, r/100.0, sp, d))
            
            total_score += r
            
            s = sp

        if memory.size() > UPDATE_SIZE:
            for _ in range(SOFT_UPDATE_TERM):
                train(mu, mu_target, q, q_target, memory, q_optim, mu_optim)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if ep % 10 == 0 and ep != 0:
            print("{} EP | REWARD : {}".format(ep, score/10))
            score = 0

main()

