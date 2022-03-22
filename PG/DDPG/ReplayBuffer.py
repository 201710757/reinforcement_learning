import numpy as np
import torch
import collections
import random

device = torch.device("cuda:0")
BUFFER_LIMIT = 50000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)
    def put(self, transition):
        self.buffer.append(transition)
    def sample(self, n):
        minibatch = random.sample(self.buffer, n)
        s, a, r, sp, d = [], [], [], [], []

        for transition in minibatch:
            _s, _a, _r, _sp, _d = transition
            s.append(_s)
            a.append([_a])
            r.append([_r])
            sp.append(_sp)
            d.append([0.0 if _d else 1.0])

        return torch.tensor(s, dtype=torch.float).to(device), torch.tensor(a, dtype=torch.float), torch.tensor(r, dtype=torch.float), torch.tensor(sp, dtype=torch.float), torch.tensor(d, dtype=torch.float)

    def size(self):
        return len(self.buffer)
