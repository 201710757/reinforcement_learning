import random

class MAB:
    def __init__(self, n=2):
        self.n = n
        self.prob = [random() for i in range(n)]


    def pull(self, action):
        if random() <= self.prob[action]:
            reward = 1
        else:
            reward =0
        return reward
