import gym
import numpy as np
import random

GAMMA = 0.99
LR = 0.1

env = gym.make('FrozenLake-v0')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.n

Q_table = np.random.randn(OBSERVATION_SPACE, ACTION_SPACE)
def pick_action(state):
    if state.var() == 0:
        return int(random.randrange(0, env.action_space.n + 1))
    else:
        return np.argmax(state)

r_list = []
episodes = 10000
for ep in range(episodes):
    obs = env.reset()
    success = 0
    while True:
        # env.render()
        a = pick_action(Q_table[obs,:])
        n_obs, reward, done, info = env.step(a)

        # Q_table[obs, a] = (1-LR)*(Q_table[obs, a]) +  LR*(reward + GAMMA * np.max(Q_table[n_obs, :]))
        Q_table[obs, a] = Q_table[obs, a] + LR*(reward + np.max(Q_table[n_obs, :]) - Q_table[obs, a])

        obs = n_obs
        success += reward
        if done:
            break
    # env.render()
    r_list.append(success)
print(r_list)
