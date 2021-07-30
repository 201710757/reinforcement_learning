import torch
import gym
import numpy as np
from gym.envs.registration import register

import random as pr
def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)
env = gym.make('FrozenLake-v3')
s = env.reset()
print(s)
print("OBS Space : {}, ACTION Space : {}".format(env.observation_space.n, env.action_space.n))
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
rList = []


for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    e = 1. / ((i // 100) + 1)
    g = 0.9
    while not done:
        # greedy act
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])      

        new_state, reward, done, info = env.step(action)

        # Update Q value
        Q[state, action] = reward + g * np.max(Q[new_state, :])
        state = new_state
        rAll += reward
    rList.append(rAll)

print("Success rate : ", str(sum(rList)/num_episodes))

import matplotlib.pyplot as plt
plt.bar(range(len(rList)), rList, color='blue')
plt.show()
