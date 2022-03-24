import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np

from ActorCritic import ActorCritic
from ParallelEnv import ParallelEnv

device = torch.device("cuda")

n_train_processes = 16
learning_rate = 0.0002
update_interval = 20
gamma = 0.98
max_train_steps = 1000000
PRINT_INTERVAL = update_interval * 100
hidden_dim = 1024

env_name = 'LunarLander-v2'#'CartPole-v1'

def test(step_idx, model):
    
    env = gym.make(env_name)
    score = 0.0
    d  = False

    freq = 10

    for _ in range(freq):
        s = env.reset()
        while not d:
            prob = model.actor(torch.FloatTensor(s).to(device), softmax_dim=0)
            a = Categorical(prob).sample().cpu().numpy()
            sp ,r, d, _ = env.step(a)

            s = sp
            score += r
        d = False
    print("{} score : {}".format(step_idx, score/freq))
    env.close()


def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)
    return torch.FloatTensor(td_target[::-1]).to(device)

if __name__ == '__main__':
    envs = ParallelEnv(n_train_processes, env_name)
    
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    model = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    step_idx = 0
    s = envs.reset()

    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(update_interval):
            prob = model.actor(torch.FloatTensor(s).to(device))
            a = Categorical(prob).sample().cpu().numpy()
            s_p, r, d, info = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(1-d)

            s = s_p

            step_idx += 1
        s_fin = torch.FloatTensor(s).to(device)
        v_fin = model.critic(s_fin).detach().clone().cpu().numpy()
        td_target = compute_target(v_fin, r_lst, mask_lst)

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.FloatTensor(s_lst).to(device).reshape(-1, input_dim)
        a_vec = torch.tensor(a_lst).to(device).reshape(-1).unsqueeze(1)
        advantage = td_target_vec - model.critic(s_vec).to(device).reshape(-1)

        pi = model.actor(s_vec, softmax_dim=1)
        pi_a = pi.gather(1, a_vec).reshape(-1)
        loss = -(torch.log(pi_a) * advantage.detach()).mean() + F.smooth_l1_loss(model.critic(s_vec).reshape(-1), td_target_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)
            #print()

    envs.close()


