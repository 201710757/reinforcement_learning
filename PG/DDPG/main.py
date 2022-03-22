import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import gym

from Network import Q, Mu
from OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise
from ReplayBuffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda:0")


LR_Q = 0.001
LR_Mu = 0.0005
GAMMA = 0.99
tau = 0.005
MAX_STEPS = 10000
batch_size = 64
UPDATE_SIZE = 2000
SOFT_UPDATE_TERM = 10

def train(mu, mu_target, q, q_target, memory, q_optim, mu_optim):
    s, a, r, sp, d = memory.sample(batch_size)
    #target = r + GAMMA * q_target(sp, torch.argmax(mu_target(sp), dim=1, keepdim=True).float()) * d
    target = r + GAMMA * q_target(sp,mu_target(sp)) * d
    critic_loss = F.mse_loss(target.detach(), q(s, a))
    q_optim.zero_grad()
    critic_loss.backward()
    q_optim.step()

    #actor_loss = -q(s, torch.argmax(mu(s), dim=1, keepdim=True).float()).mean()
    actor_loss = -q(s, mu(s)).mean()
    mu_optim.zero_grad()
    actor_loss.backward()
    mu_optim.step()

def soft_update(net, net_grad):
    for p_t, p in zip(net_grad.parameters(), net.parameters()):
        p_t.data.copy_(tau * p.data + (1.0 - tau) * p_t.data)

def main():
    env_name = 'Pendulum-v0'
    writer = SummaryWriter("runs/"+ env_name + "_" + time.ctime(time.time()))

    env = gym.make(env_name)

    input_dim = env.observation_space.shape[0]
    output_dim = 1#env.action_space.n
    memory = ReplayBuffer()

    q, q_target = Q(input_dim).to(device), Q(input_dim).to(device)
    mu, mu_target = Mu(input_dim, output_dim).to(device), Mu(input_dim, output_dim).to(device)

    q_target.load_state_dict(q.state_dict())
    mu_target.load_state_dict(mu.state_dict())

    q_optim = optim.Adam(q.parameters(), lr = LR_Q)
    mu_optim = optim.Adam(mu.parameters(), lr = LR_Mu)
    N = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    total_score = 0
    for ep in range(MAX_STEPS):
        s = env.reset()
        d = False
        r = 0
        while not d:
            a = mu(torch.FloatTensor(s).to(device))
            noise = N()[0]
            #a = torch.argmax(a * noise)#N()[0]
            a = a.item() + noise
            #print(a)
            sp, r, d, _ = env.step([a])
            memory.put((s, a, (r+8)/8, sp, d))
            
            total_score += r
            
            s = sp

        if memory.size() > UPDATE_SIZE:
            for _ in range(SOFT_UPDATE_TERM):
                train(mu, mu_target, q, q_target, memory, q_optim, mu_optim)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if ep % 10 == 0 and ep != 0:
            writer.add_scalar("Model - Average 10 steps", total_score/10, ep)
            print("{} EP | REWARD : {}".format(ep, total_score/10))
            total_score = 0

main()

