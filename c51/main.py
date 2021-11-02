from collections import deque
import random
from ReplayMemory import ReplayMemory, Transition
from C51 import C51
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gym
import numpy as np
import math
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 2000

env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

REPLAY_MEMORY = 10000
memory = ReplayMemory(REPLAY_MEMORY)

c51_net = C51(n_states, 51, n_actions)

steps_done = 0

num_episodes = 20000
for i_episodes in range(num_episodes):
    state = env.reset()
    state = np.array(state)
    for t in count():
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1*t / EPS_DECAY)

        action = c51_net.pick_action(state, eps_threshold)
        print("ACTION : ", action)
        
        n_s, r, d, _ = env.step(action)
        n_s = np.array(n_s)
        c51_net.store_transition(state, action, r, n_s, d)

        state = n_s
        
        if d:
            break

    if len(c51_net.memory_counter % 1000 == 0 and c51_net.memory_counter > 1):
        c51_net.learn()

    if i_episodes % 1000 == 0:
        print("EPI {} ".format(i_episodes))

def bot_play(agent):

    state = env.reset()
    total_reward = 0

    while True:
        env.render()
        action = agent.pick_action(state, -1)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            print("Total score: {}".format(total_reward))
            break
bot_play(c51_net)