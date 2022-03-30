import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from collections import namedtuple

import gym
from ActorCriticCNN import A2C_LSTM
from two_step import TwoStepTask

Rollout = namedtuple('Rollout', ('state', 'action', 'reward', 'timestep', 'done', 'policy', 'value'))

hidden_dim = 512
MAX_EP = 10000
GAMMA = 0.9
TEST_EP = 10

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda")
        
        self.env = gym.make('Pong-v0')# TwoStepTask()
        # self.agent = A2C_LSTM(self.env.feat_size, hidden_dim, self.env.num_actions).to(self.device)
        self.agent = A2C_LSTM(self.env.observation_space.shape[0], hidden_dim, self.env.action_space.n).to(self.device)
        self.optim = optim.RMSprop(self.agent.parameters(), lr=7.e-4)

        self.val_coeff = 0.05
        self.entropy_coeff = 0.05
        self.max_grad_norm = 999.
        self.switch_p = 0.1
        self.start_episode = 0

        self.writer = SummaryWriter('runs/')
        # self.save_path = 
    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).reshape(1,80,80)

    def run_episode(self, episode):
        done = False
        total_reward = 0
        p_action, p_reward, timestep = [0]*self.env.action_space.n, 0, 0

        state = self.env.reset()
        mem_state = self.agent.get_init_states()

        buffer = []
        
        while not done:
            # self.env.possible_switch(switch_p=self.switch_p)
            state = self.prepro(state)

            action_dist, val_estimate, mem_state = self.agent((
                    torch.tensor([state], device=self.device).float(),
                    torch.tensor([p_action], device=self.device).float(),
                    torch.tensor([[p_reward]], device=self.device).float(),
                    torch.tensor([[timestep]], device=self.device).float(),
                    mem_state
            ))

            action_cat = Categorical(action_dist.squeeze())
            action = action_cat.sample()
            action_onehot = np.eye(self.env.action_space.n)[action]

            new_state, reward, done, _ = self.env.step(int(action))
            timestep += 1
            buffer += [Rollout(state, action_onehot, reward, timestep, done, action_dist, val_estimate)]

            state = new_state
            #state = self.prepro(state)
            p_reward = reward
            p_action = action_onehot

            total_reward += reward

        state = self.prepro(state)
        _, val_estimate, _ = self.agent((
                torch.tensor([state], device=self.device).float(),
                torch.tensor([p_action], device=self.device).float(),
                torch.tensor([[p_reward]], device=self.device).float(),
                torch.tensor([[timestep]], device=self.device).float(),
                mem_state
        ))
        
        
        buffer += [Rollout(None, None, None, None, None, None, val_estimate)]

        return total_reward, buffer

    def a2c_loss(self, buffer, gamma, lambd=1.0):
        _, _, _, _, _, _, last_value = buffer[-1]
        returns = last_value.data
        advantages = 0

        all_returns = torch.zeros(len(buffer)-1, device=self.device)
        all_advantages = torch.zeros(len(buffer)-1, device=self.device)

        for t in reversed(range(len(buffer)-1)):
            _, _, reward, _, done, _, value = buffer[t]
            _, _, _, _, _, _, next_value = buffer[t+1]

            mask = ~done

            returns = reward + returns * gamma * mask

            deltas = reward + next_value.data * gamma * mask - value.data
            advantages = advantages * gamma * lambd * mask + deltas

            all_returns[t] = returns
            all_advantages[t] = advantages

        batch = Rollout(*zip(*buffer))

        policy = torch.cat(batch.policy[:-1], dim=1).squeeze().to(self.device)
        action = torch.tensor(batch.action[:-1], device=self.device)
        values = torch.tensor(batch.value[:-1], device=self.device)

        logits = (policy * action).sum(1)
        policy_loss = -(torch.log(logits) * all_advantages).mean()
        value_loss = 0.5 * (all_returns - values).pow(2).mean()
        entropy_reg = -(policy * torch.log(policy)).mean()

        loss = self.val_coeff * value_loss + policy_loss - self.entropy_coeff * entropy_reg

        return loss

    def train(self, max_episodes, gamma):
        total_rewards = np.zeros(max_episodes)
        
        for ep in range(max_episodes):
            reward, buffer = self.run_episode(ep)

            self.optim.zero_grad()
            loss = self.a2c_loss(buffer, gamma)
            loss.backward()

            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optim.step()

            total_rewards[ep] = reward

            print("Ep : {} | Reward : {} | Mean Reward : {}".format(ep, reward, total_rewards[max(0, ep-10):ep+1].mean()))

    def _test(self, num_ep):
        self.env.reset_transition_count()
        self.agent.eval()
        total_rewards = np.zeros(num_ep)

        for ep in range(num_ep):
            reward, _ = self.run_episode(ep)
            total_rewards[ep] = reward
            print("Ep : {} | Reward : {} | Mean Reward : {}".format(ep, reward, total_rewards[max(0, ep-10):ep+1].mean()))
        self.env.plot('plots/res')

    def test(self, num_ep):
        self.agent.eval()
        
        for ep in range(num_ep):
            reward, _ = self.run_episode(ep)
            total_rewards[ep] = reward
            print("EP : {} | Reward : {} | Mean Reward : {}".format(ep, reward, total_reward[max(0, ep-10):ep+1].mean()))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(MAX_EP, GAMMA)
    trainer.test(TEST_EP)



