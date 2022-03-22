import torch
import torch.nn as nn


device = torch.device("cuda:0")

class Mu(nn.Module):
    def __init__(self, input_dim):
        super(Mu, self).__init__()
        
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        m = torch.tanh(self.mu(x))

        return m




class Q(nn.Module):
    def __init__(self, input_dim):
        super(Q, self).__init__()

        self.input_dim = input_dim

        self.fc_s = nn.Linear(self.input_dim, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_Q = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, s, a):
        _s = F.relu(self.fc_s(s))
        _a = F.relu(self.fc_a(a))
        sa = torch.cat([h1, h2], dim=1)

        Q = F.relu(self.fc_Q(sa))
        Q = self.out(Q)

        return Q
