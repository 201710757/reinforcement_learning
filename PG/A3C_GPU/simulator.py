import gym
import torch
import numpy as np

# Factory to create a simulator object
def _get_simulator():
    return gym.make('PongNoFrameskip-v4')

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I==144]=0
    I[I==109]=0
    I[I!=0]=1
    return I.astype(np.float).ravel()

# Convert state object to tensor (Edit this function depending upon the simulator)
def _state_to_tensor(state):
    state = prepro(state)
    return torch.tensor(state).float().flatten()
