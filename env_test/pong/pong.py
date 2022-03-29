import gym
import time
import numpy as np
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


env = gym.make('Pong-v0')

s = env.reset()

d = False
tot = 0
while not d:
    s_ = prepro(s)
    if tot == 0:
        print(s_.shape)
    s, r, d, _ = env.step(env.action_space.sample())
    tot += r
    env.render()
env.render()
time.sleep(10.)

print("REWARD : ", tot)
