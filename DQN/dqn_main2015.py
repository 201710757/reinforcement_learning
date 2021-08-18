import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from DQN import qnet
from ReplayMemory import ReplayMemory
import math 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

GAMMA = 0.99
LR = 0.01
REPLAY_MEMORY = 10000
BATCH_SIZE = 128
TARGET_UPDATE_FREQUENCY = 20

env = gym.make('CartPole-v1')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]

main_model = qnet(OBSERVATION_SPACE, ACTION_SPACE).to(device)
main_model.eval()
target_model = qnet(OBSERVATION_SPACE, ACTION_SPACE).to(device)
target_model.load_state_dict(main_model.state_dict())
target_model.eval()

optimizer = optim.Adam(main_model.parameters(), lr=LR)


def train_minibatch(minibatch):
    state_arr = torch.cat([torch.tensor([x[0]]).float() for x in minibatch])
    action_arr = torch.cat([torch.tensor([x[1]]) for x in minibatch])
    reward_arr = torch.cat([torch.tensor([x[2]]) for x in minibatch]).to(device)
    next_state_arr = torch.cat([torch.tensor([x[3]]).float() for x in minibatch])
    done_arr = torch.cat([torch.tensor([x[4]]) for x in minibatch]).to(device)

    Q = main_model(state_arr)
    with torch.no_grad():
        next_state_value = target_model(next_state_arr)
    # n_s_b = torch.zeros(len(Q)).to(device)
    # n_s_b = (reward_arr + GAMMA * next_state_value.max(1)[0] * (done_arr==False)) + ((done_arr==True) * reward_arr)
    
    # Q[np.arange(len(Q)), action_arr] = torch.tensor(reward_arr + GAMMA * next_state_value.max(1)[0] * (done_arr != True)).to(device)\
    #     + torch.tensor((done_arr == True) * reward_arr).to(device)
    Q[np.arange(len(Q)), action_arr] = torch.tensor((reward_arr + GAMMA * next_state_value.max(1)[0] * (done_arr==False)) + ((done_arr==True) * reward_arr)).to(device)
    x_batch = main_model(state_arr)

    criterion = nn.MSELoss()
    loss = criterion(x_batch, Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def pick_action(state):
    return torch.argmax(state).item()

cnt_list = []
episodes = 1500
replay_buffer = ReplayMemory(REPLAY_MEMORY)

# steps_done = 0
for ep in range(episodes):
    obs = env.reset()

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*ep)
    cnt = 0

    while True:
        sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        # steps_done += 1

        cnt += 1
        exploration = random.uniform(0, 1)
        Q = main_model(torch.tensor(obs).float())
        if exploration > exploration_rate:
            a = pick_action(Q)
        else:
            a = env.action_space.sample()

        n_obs, reward, done, info = env.step(a)
        if done:
            reward = -100
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

    if ep % TARGET_UPDATE_FREQUENCY == 0:
        target_model.load_state_dict(main_model.state_dict())
    print("ep : {} / step count : {} / E : {} ".format(ep, cnt, exploration_rate))

    # enough
    if len(cnt_list) > 15 and np.mean(cnt_list[-10:]) > 500:
        break
    

reward_sum = 0
for i in range(10):
    obs = env.reset()
    step_cnt = 0

    while True:
        env.render()

        Q = main_model(torch.tensor(obs).float())
        action = pick_action(Q)

        n_obs, reward, done, _ = env.step(action)
        step_cnt += 1
        if done:
            print("{}th Step : {}".format(i, step_cnt))
            break
