import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.n

Q_table = np.random.randn(OBSERVATION_SPACE, ACTION_SPACE)
def pick_action(state):
    if state.var() == 0:
        return int(random.randrange(0, env.action_space.n + 1))
    else:
        return np.argmax(state)

success = 0
episodes = 1000
for ep in range(episodes):
    obs = env.reset()

    while True:
        a = pick_action(Q_table[obs,:])
        n_obs, reward, done, info = env.step(a)

        Q_table[obs, a] = reward + np.max(Q_table[n_obs, :])
        obs = n_obs
        success += reward
        if done:
            break
print("Success Rate : {}".format(success))
