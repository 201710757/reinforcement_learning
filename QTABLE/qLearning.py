import gym
import numpy as np
import random
import torch

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

# Q_table = np.zeros([OBSERVATION_SPACE, ACTION_SPACE])
arr = np.zeros([OBSERVATION_SPACE, ACTION_SPACE])
Q_table = torch.tensor(arr, device=device)
def pick_action(state):
    if state.var() == 0:
        return int(random.randrange(0, env.action_space.n))
    else:
        return np.argmax(state)


r_list = []
episodes = 100000
success=0
for ep in range(episodes):
    obs = env.reset()
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*ep)

    while True:
        # env.render()
        exploration = random.uniform(0, 1)
        if exploration > exploration_rate:
            a = pick_action(Q_table[obs,:])
        else:
            a = env.action_space.sample()
        n_obs, reward, done, info = env.step(a)

        Q_table[obs, a] = (1-LR)*(Q_table[obs, a]) +  LR*(reward + GAMMA * np.max(Q_table[n_obs, :]))
        # Q_table[obs, a] = Q_table[obs, a] + LR*(reward + np.max(Q_table[n_obs, :]) - Q_table[obs, a])

        obs = n_obs
        success += reward
        if done:
            break
    if ep % 10000 == 0:
        print("middle score : {}%".format((success / (ep+1))*100))
    # env.render()
    
print("Total score : {}%".format((success / episodes)*100))
