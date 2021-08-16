import gym
import numpy as np

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

import random
from DQN import DQN
from ReplayMemory import ReplayMemory

REPLAY_MEMORY = 10000
BATCH_SIZE = 128
TARGET_UPDATE_FREQUENCY = 10

env = gym.make('CartPole-v0')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]

main_model = DQN(OBSERVATION_SPACE, ACTION_SPACE)

target_model = DQN(OBSERVATION_SPACE, ACTION_SPACE)
target_model.model.set_weights(main_model.model.get_weights())


def train_minibatch(minibatch):
    print(len(minibatch))
    for state, action, reward, next_state, done in minibatch:
        #print(state[0], action, reward, next_state[0], done)
        target = target_model.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            # print(next_state)
            # print("AA  :   ", target_model.model.predict(next_state))
            Q_ = max(target_model.model.predict(next_state)[0])
            # print(Q_)
            target[0][action] = reward + Q_ * target_model.gamma
        main_model.model.fit(state, target, epochs=1, verbose=0)
        


cnt_list = []
episodes = 200
replay_buffer = ReplayMemory(REPLAY_MEMORY)
for ep in range(episodes):
    obs = env.reset().reshape(-1, OBSERVATION_SPACE)
    cnt = 0
    
    while True:
        cnt += 1
        a = main_model.act(obs)

        n_obs, reward, done, info = env.step(a)
        if done:
            reward = -100
        n_obs = n_obs.reshape(-1, OBSERVATION_SPACE)
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
        target_model.model.set_weights(main_model.model.get_weights())
    print("ep : {} / step count : {}  ".format(ep, cnt))

    # enough
    if len(cnt_list) > 15 and np.mean(cnt_list[-10:]) > 500:
        break

main_model.save("model_keras.model")  

reward_sum = 0
for i in range(10):
    obs = env.reset()
    while True:
        env.render()

        action = main_model.model.predict(obs)
        n_obs, reward, done, _ = env.step(action)

        reward_sum += reward
        if done:
            break
print("mean score: {}".format(reward_sum / 10))
