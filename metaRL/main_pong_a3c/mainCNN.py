import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from collections import namedtuple

import torch.multiprocessing as mp
import gym
from ActorCriticCNN import A2C_LSTM
from ParallelEnv import ParallelEnv
import time


Rollout = namedtuple('Rollout', ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value'))

lambd=0.99#1.0
hidden_dim = 256
MAX_EP = 1000000
GAMMA = 0.9
TEST_EP = 10
#class Trainer:
#    def __init__(self):
device = torch.device("cuda")

env_name = 'Pong-v0'
process_num = 3

#env = gym.make('Pong-v0')#envs = ParallelEnv(n_train_processes, env_name)# gym.make('Pong-v0')
#local_policy = A2C_LSTM(env.observation_space.shape[0], hidden_dim, self.env.action_space.n).to(device)
#self.optim = optim.RMSprop(local_policy.parameters(), lr=1.e-4)
LR = 1.e-4
val_coeff = 0.05
entropy_coeff = 0.05
max_grad_norm = 999.
switch_p = 0.1
start_episode = 0

writer = SummaryWriter('runs/' + env_name + "_" + time.ctime(time.time()))
# self.save_path = 
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).reshape(1,80,80)

def train(g_policy, rank):
    total_rewards = np.zeros(MAX_EP)
    env = gym.make(env_name)

    local_policy = A2C_LSTM(env.observation_space.shape[0], hidden_dim, env.action_space.n).to(device)
    local_policy.load_state_dict(g_policy.state_dict())

    local_optimizer = optim.Adam(g_policy.parameters(), lr = LR)

    mem_state = local_policy.get_init_states()
    for ep in range(MAX_EP):
        done = False
        total_reward = 0
        p_action, p_reward, timestep = [0]*env.action_space.n, 0, 0

        state = env.reset()
        
        # mem_state = local_policy.get_init_states()

        buffer = []
        
        while not done:
            mem_state = (mem_state[0].detach(), mem_state[1].detach())
            # env.possible_switch(switch_p=self.switch_p)
            state = prepro(state)

            action_dist, val_estimate, mem_state = local_policy((
                    torch.tensor([state], device=device).float(),
                    torch.tensor([p_action], device=device).float(),
                    torch.tensor([[p_reward]], device=device).float(),
                    #torch.tensor([[timestep]], device=device).float(),
                    mem_state
            ))

            action_cat = Categorical(action_dist.squeeze())
            action = action_cat.sample()
            action_onehot = np.eye(env.action_space.n)[action]

            new_state, reward, done, _ = env.step(action)
            timestep += 1
            buffer += [Rollout(state, action_onehot, reward, timestep, done, action_dist, val_estimate)]

            state = new_state
            #state = prepro(state)
            p_reward = reward
            p_action = action_onehot

            total_reward += reward

        state = prepro(state)
        mem_state = (mem_state[0].detach(), mem_state[1].detach())
        _, val_estimate, _ = local_policy((
                torch.tensor([state], device=device).float(),
                torch.tensor([p_action], device=device).float(),
                torch.tensor([[p_reward]], device=device).float(),
                #torch.tensor([[timestep]], device=device).float(),
                mem_state
        ))
        
        
        buffer += [Rollout(None, None, None, None, None, None, val_estimate)]

        #return total_reward, buffer

    #def a2c_loss(self, buffer, GAMMA, lambd=1.0):
        _, _, _, _, _, _, last_value = buffer[-1]
        returns = last_value.data
        advantages = 0

        all_returns = torch.zeros(len(buffer)-1, device=device)
        all_advantages = torch.zeros(len(buffer)-1, device=device)

        for t in reversed(range(len(buffer)-1)):
            _, _, reward, _, done, _, value = buffer[t]
            _, _, _, _, _, _, next_value = buffer[t+1]

            mask = ~done

            returns = reward + returns * GAMMA * mask

            deltas = reward + next_value.data * GAMMA * mask - value.data
            advantages = advantages * GAMMA * lambd * mask + deltas

            all_returns[t] = returns
            all_advantages[t] = advantages

        batch = Rollout(*zip(*buffer))

        policy = torch.cat(batch.policy[:-1], dim=1).squeeze().to(device)
        action = torch.tensor(batch.action[:-1], device=device)
        values = torch.tensor(batch.value[:-1], device=device)

        logits = (policy * action).sum(1)
        policy_loss = -(torch.log(logits) * all_advantages).mean()
        value_loss = 0.5 * (all_returns - values).pow(2).mean()
        entropy_reg = -(policy * torch.log(policy)).mean()

        loss = val_coeff * value_loss + policy_loss - entropy_coeff * entropy_reg

        #return loss

    #def train(self, max_episodes, GAMMA, global_policy):
        #total_rewards = np.zeros(max_episodes)
        
        #for ep in range(max_episodes):
        #reward, buffer = self.run_episode(ep, global_policy)

        local_optimizer.zero_grad()
        #loss = self.a2c_loss(buffer, GAMMA)
        loss.backward()

        for g_param, l_param in zip(g_policy.parameters(), local_policy.parameters()):
            g_param._grad = l_param._grad
        local_optimizer.step()
        local_policy.load_state_dict(g_policy.state_dict())

        #if self.max_grad_norm > 0:
        #    grad_norm = nn.utils.clip_grad_norm_(local_policy.parameters(), self.max_grad_norm)
        #self.optim.step()

        total_rewards[ep] = total_reward
        if rank == 1:
            writer.add_scalar("Avg 10 ep", total_rewards[max(0, ep-10):ep+1].mean(), ep)
            print("Ep : {} | Reward : {} | Mean Reward : {}".format(ep, total_reward, total_rewards[max(0, ep-10):ep+1].mean()))


def test(num_ep):
    local_policy.eval()
    total_rewards = []
    for ep in range(num_ep):
        reward, _ = self.run_episode(ep)
        total_rewards.append(reward)
        print("EP : {}/{} | Reward : {} | Mean Reward : {}".format(ep, MAX_EP, reward, total_reward[max(0, ep-10):ep+1].mean()))

if __name__ == "__main__":
    env = gym.make('Pong-v0')
    s = env.reset()

    global_policy = A2C_LSTM(env.observation_space.shape[0], hidden_dim, env.action_space.n).to(device)
    global_policy.share_memory()

    processes = []
    
    global_policy.train()

    mp.set_start_method('spawn')
    
    for rank in range(process_num):
        p = mp.Process(target=train, args=(global_policy, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # trainer = Trainer()
    # trainer.train(MAX_EP, GAMMA)
    # trainer.test(TEST_EP)



