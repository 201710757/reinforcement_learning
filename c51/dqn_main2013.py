import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from DQN import qnet
from ReplayMemory import ReplayMemory

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

GAMMA = 0.99
LR = 0.001
REPLAY_MEMORY = 50000
BATCH_SIZE = 64

env = gym.make('CartPole-v0')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]

model = qnet(OBSERVATION_SPACE, ACTION_SPACE).to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)

def pick_action(state):
    return torch.argmax(state).item()


def train_minibatch(minibatch):
    state_arr = torch.cat([torch.tensor([x[0]]).float() for x in minibatch])
    action_arr = torch.cat([torch.tensor([x[1]]) for x in minibatch])
    reward_arr = torch.cat([torch.tensor([x[2]]) for x in minibatch])
    next_state_arr = torch.cat([torch.tensor([x[3]]).float() for x in minibatch])
    done_arr = torch.cat([torch.tensor([~x[4]]) for x in minibatch])

    x_batch = model(state_arr)
    y_batch = model(state_arr)

    with torch.no_grad():
        next_state_value = model(next_state_arr)
    Q_target = torch.tensor(reward_arr + done_arr * GAMMA * torch.max(next_state_value).to(device).item()).to(device)
    y_batch[np.arange(len(x_batch)), action_arr] = Q_target

    criterion = nn.MSELoss()
    loss = criterion(x_batch, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



cnt_list = []
episodes = 100000
replay_buffer = ReplayMemory(REPLAY_MEMORY)
for ep in range(episodes):
    obs = env.reset()
    # cartpole doesnt need one hot encoding
    # obs = np.eye(OBSERVATION_SPACE)[obs]
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*ep)
    cnt = 0

    while True:
        cnt += 1
        exploration = random.uniform(0, 1)
        Q = model(torch.tensor(obs).float())
        if exploration > exploration_rate:
            a = pick_action(Q)
        else:
            a = env.action_space.sample()

        n_obs, reward, done, info = env.step(a)
        if done:
            reward = -1
        replay_buffer.push(obs, a, reward, n_obs, done)
        obs = n_obs

        if len(replay_buffer) > BATCH_SIZE:
            minibatch = replay_buffer.sample(BATCH_SIZE)
            train_minibatch(minibatch)
        if len(replay_buffer) >= REPLAY_MEMORY:
            replay_buffer.pop_left()
        
        if done:
            cnt_list.append(cnt)
            break
    print("ep : {} / step count : {} / E : {} ".format(ep, cnt, exploration_rate))

    # enough
    if len(cnt_list) > 15 and np.mean(cnt_list[-10:]) > 500:
        break
    
reward_sum = 0
obs = env.reset()
while True:
    env.render()

    Q = model(torch.tensor(obs).float())
    action = pick_action(Q)

    n_obs, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break