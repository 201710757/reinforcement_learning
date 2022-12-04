import gym
import numpy as np

env = gym.make('FrozenLake-v1')
OBS_N = env.observation_space.n
ACTION_N = env.action_space.n

max_ep = 1000
end_point = 1e-10
GAMMA = 0.99

Q = np.zeros([OBS_N])
policy = np.zeros(OBS_N)

for ep in range(max_ep):
    new_Q = np.copy(Q)

    for s in range(OBS_N):
        Q_val = [sum([prob*(r + GAMMA * new_Q[s_]) for prob, s_, r, _ in env.P[s][a]]) for a in range(ACTION_N)]
        
        Q[s] = np.max(Q_val)
        policy[s] = np.argmax(np.array(Q_val))


    if np.sum(np.fabs(new_Q - Q)) <= end_point:
        break
print(Q)
print()
print(policy)

