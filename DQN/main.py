from collections import deque
import random
from ReplayMemory import ReplayMemory, Transition
from DQN import DQN
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
TARGET_UPDATE = 10

env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1*steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net.predict(state)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))


    state_batch = torch.from_numpy(np.stack([x for x in batch.state])).float().to(device)
    action_batch = torch.from_numpy(np.vstack([x for x in batch.action])).long().to(device)
    reward_batch = torch.from_numpy(np.vstack([x for x in batch.reward])).float().to(device)
    next_state_batch = torch.from_numpy(np.stack([x for x in batch.next_state])).float().to(device)
    done_batch = torch.from_numpy(np.vstack([x for x in batch.done]).astype(np.int8)).float().to(device)


    q_out = policy_net(state_batch)
    q_a = q_out.gather(1, action_batch)
    max_q_p = target_net(next_state_batch).max(1)[0].unsqueeze(1)
    target = reward_batch + GAMMA*max_q_p*done_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_a, target)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


    '''
    state_batch = torch.stack([x for x in batch.state])
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack([torch.tensor(x).to(device) for x in batch.reward])
    next_state_batch = torch.stack([x for x in batch.next_state])
    done_batch = torch.stack([torch.tensor(x).to(device) for x in batch.done])

    print(state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape, done_batch.shape)



    batch = Transition(*zip(*transitions))
    state_batch = np.vstack(batch.state)
    action_batch = np.vstack(batch.action)
    reward_batch = np.vstack(batch.reward)
    next_state_batch = np.vstack(batch.next_state)
    done_batch = np.vstack(batch.done)
    '''

'''
    X = state_batch
    Q_target = reward_batch + GAMMA * target_net.predict(next_state_batch)#*~done_batch
    y = policy_net.predict(state_batch)
    print(y)
    y[np.arange(len(state_batch)), action_batch] = Q_target


    criterion = nn.SmoothL1Loss()
    loss = criterion(X, y)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
'''

num_episodes = 1000
for i_episodes in range(num_episodes):
    state = env.reset()

    for t in count():
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1*t / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                action = policy_net.predict(state)
        else:
            action = torch.tensor(env.action_space.sample(), device=device)
            # action = env.action_space.sample()

        # action = np.argmax(action)
        n_s, r, d, _ = env.step(action.item())
        
        if d:
            r = -1
        memory.push(state, action.cpu(), r, n_s, d)
        state = n_s
        
        optimize_model()
        if d:
            break
    
    if i_episodes % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

def bot_play(agent):
    """Runs a single episode with rendering and prints a reward
    Args:
        mainDQN (dqn.DQN): DQN Agent
    """
    state = env.reset()
    total_reward = 0

    while True:
        env.render()
        action = agent.predict(state)
        state, reward, done, _ = env.step(action.item())
        total_reward += reward
        if done:
            print("Total score: {}".format(total_reward))
            break
bot_play(policy_net)