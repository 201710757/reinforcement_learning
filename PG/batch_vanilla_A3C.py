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
from ActorCritic_sep import Actor
from ActorCritic_sep import Critic
import torch.multiprocessing as mp

device = torch.device("cuda")
# env_name = 'CartPole-v1'
env_name = 'LunarLander-v2'
env = gym.make(env_name)

writer = SummaryWriter("runs/"+ env_name)

input_dim = env.observation_space.shape[0]
hidden_dim = 1024
output_dim = env.action_space.n
LR = 1e-3
MAX_EP = 5000

# 4 : memory error
process_num = 3
batch_size = 5
GAMMA = 0.99
MAX_STEP = 20000

def train(g_actor, g_critic, model_num):
    env = gym.make(env_name)
    local_actor = Actor(input_dim, hidden_dim, output_dim).to(device)
    local_critic = Critic(input_dim, hidden_dim, output_dim).to(device)

    local_actor.load_state_dict(g_actor.state_dict())
    local_critic.load_state_dict(g_critic.state_dict())

    local_optimizer_actor = optim.Adam(g_actor.parameters(), lr = LR)
    local_optimizer_critic = optim.Adam(g_critic.parameters(), lr = LR)

    train_reward = []
    batch = []
    for ep in range(MAX_EP):
        ep_reward = 0

        s = env.reset()
        d = False
        step = 0

        while (not d) and (step < MAX_STEP):
            _s = torch.FloatTensor(s).to(device)
            action_prob = local_actor(_s)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()

            s_p, r, d, _ = env.step(action.item())
            
            batch.append([s, r, s_p, action_prob[action], d])
            
            if len(batch) >= batch_size:
                s_buf = []
                r_buf = []
                d_buf = []
                s_p_buf = []
                prob_buf = []

                for item in batch:
                    s_buf.append(item[0])
                    r_buf.append(item[1])
                    d_buf.append(item[4])
                    s_p_buf.append(item[2])
                    prob_buf.append(item[3])
                s_buf = torch.FloatTensor(s_buf).to(device)
                r_buf = torch.FloatTensor(r_buf).unsqueeze(1).to(device)
                d_buf = torch.FloatTensor(d_buf).unsqueeze(1).to(device)
                s_p_buf = torch.FloatTensor(s_p_buf).to(device)

                v_s = local_critic(s_buf)
                v_s_p = local_critic(s_p_buf)

                Q = r_buf + GAMMA * v_s_p.detach()*d_buf
                A = Q - v_s

                local_optimizer_critic.zero_grad()
                critic_loss = F.mse_loss(v_s, Q.detach())
                critic_loss.backward()
                for g_param, l_param in zip(g_critic.parameters(), local_critic.parameters()):
                    g_param._grad = l_param._grad
                local_optimizer_critic.step()

                local_optimizer_actor.zero_grad()
                actor_loss = 0
                for idx, prob in enumerate(prob_buf):
                    actor_loss += -A[idx].detach() * torch.log(prob)
                actor_loss /= len(prob_buf)
                actor_loss.backward()
                
                for g_param, l_param in zip(g_actor.parameters(), local_actor.parameters()):
                    g_param._grad = l_param._grad
                local_optimizer_actor.step()

                local_actor.load_state_dict(g_actor.state_dict())
                local_critic.load_state_dict(g_critic.state_dict())

                batch = []
            
            s = s_p
            ep_reward += r
            step += 1

        train_reward.append(ep_reward)
            
        if ep % 100 == 0 and model_num == 0:
            writer.add_scalar("Model - Average 10 steps", np.mean(train_reward[-100:]), ep)

        if ep % 100 == 0:
            print("MODEL{} - EP : {} | Mean Reward : {}".format(model_num, ep, np.mean(train_reward[-100:])))

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)


if __name__ == "__main__":
    

    global_actor = Actor(input_dim, hidden_dim, output_dim).to(device)
    global_critic = Critic(input_dim, hidden_dim, output_dim).to(device)
    global_actor.share_memory()
    global_critic.share_memory()

    processes = []


    global_actor.apply(init_weights)
    global_critic.apply(init_weights)

    # global_optimizer = optim.Adam(global_policy.parameters(), lr = LR)
    global_actor.train()
    global_critic.train()

    try:
        mp.set_start_method('spawn')
        print("MP start method:",mp.get_start_method())
    except:
        pass

    for rank in range(process_num):
        p = mp.Process(target=train, args=(global_actor, global_critic,rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


