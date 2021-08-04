import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from QNETWORK import qnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
GAMMA = 0.99
LR = 0.1

env = gym.make('FrozenLake-v0')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.n

model = qnet(OBSERVATION_SPACE, ACTION_SPACE).to(device)
model.eval()

optimizer = optim.SGD(model.parameters(), lr=LR)

def pick_action(state):
    return torch.argmax(state).item()


r_list = []
episodes = 100000
success=0
for ep in range(episodes):
    obs = env.reset()
    # simple one hot encoding
    obs = np.eye(OBSERVATION_SPACE)[obs]
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*ep)

    while True:
        exploration = random.uniform(0, 1)
        Q = model(torch.tensor(obs).float())
        if exploration > exploration_rate:
            a = pick_action(Q)
        else:
            a = env.action_space.sample()
        n_obs, reward, done, info = env.step(a)
        n_obs = np.eye(OBSERVATION_SPACE)[n_obs]

        if done:
            Q[a] = torch.tensor(reward)
        else:
            with torch.no_grad():
                Q_next_state_value = model(torch.tensor(n_obs).float())
            Q[a] = torch.tensor(reward + GAMMA * torch.max(Q_next_state_value).to(device).item())

        criterion = nn.MSELoss()
        loss = criterion(model(torch.tensor(obs).float()), Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = n_obs
        success += reward

        if done:
            break
    if ep % 500 == 0:
        print("ep : {} / middle score : {}%".format(ep, (success / (ep+1))*100))
    
print("Total score : {}%".format((success / episodes)*100))
