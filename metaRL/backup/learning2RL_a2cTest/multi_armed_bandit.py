import random
import torch
class MAB:
    def __init__(self, n=2):
        #self.prob = [0.1, 0.9]
        self.prob = [0.3, 0.7]
        #self.prob = self.prob[::-1] if random.random() < 0.5 else self.prob
    def pull(self, action):
        if random.random() <= self.prob[action]:
            reward = 1.0
        else:
            reward =0.0
        return reward
