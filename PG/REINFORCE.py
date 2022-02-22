""" Monte-Carlo Policy Gradient """
from __future__ import print_function
from tqdm import tqdm
import gym
import numpy as np

import MultiPro
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable

MAX_EPISODES = 1500
MAX_TIMESTEPS = 1000

ALPHA = 3e-5
GAMMA = 0.99

class reinforce(nn.Module):

    def __init__(self):
        super(reinforce, self).__init__()
        # policy network
        self.fc1 = nn.Linear(4, 512)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to('cuda')
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def get_action(self, state):
        state = Variable(state).to('cuda')
        state = torch.unsqueeze(state, 0)
        probs = self.forward(state)
        probs = torch.squeeze(probs, 0)
        action = torch.multinomial(probs, 1)
        #print(action.item())
        #print("ACITON : ", action)
        action = [act.item() for act in action]
        #action = action[0]
        return action

    def pi(self, s, a):
        # s = torch.Tensor(s).to('cuda')
        # print("S : ", s)
        # s = Variable(s)
        # print("S : ", s)
        s = torch.reshape(s, (1, -1))
        probs = self.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]

    def update_weight(self, states, actions, rewards, optimizer):
        G = Variable(torch.Tensor([0])).to('cuda')
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = Variable(torch.Tensor([r_tt])).to('cuda') + GAMMA * G
            loss = (-1.0) * G * torch.log(self.pi(s_t, a_t))
            # update policy parameter \theta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():

    #env = gym.make('CartPole-v1')
    worker = 4
    env_name = 'CartPole-v1'
    env = MultiPro.SubprocVecEnv([lambda: gym.make(env_name) for i in range(worker)])

    agent = reinforce().to('cuda')
    optimizer = optim.Adam(agent.parameters(), lr=ALPHA)
    # max_ep = tqdm(range(MAX_EPISODES))

    for i_episode in range(MAX_EPISODES):#max_ep:
        state = env.reset()

        states = []
        actions = []
        rewards = [0]   # no reward at t = 0

        for timesteps in range(MAX_TIMESTEPS):
            state = torch.Tensor(state).to('cuda')
            action = agent.get_action(state)

            #states.append(state)
            for i in range(len(state)):
                states.append(state[i])
            for i in range(len(action)):
                actions.append(action[i])
            #actions.append(action)

            state, reward, done, _ = env.step(action)

            #rewards.append(reward)
            for i in range(len(reward)):
                rewards.append(reward[i])

            if done.any():
                print("Episode {} finished after {} timesteps / REWARD : {}".format(i_episode, timesteps+1, sum(rewards)))
                break

        agent.update_weight(states, actions, rewards, optimizer)

    env.close()

if __name__ == "__main__":
    main()
