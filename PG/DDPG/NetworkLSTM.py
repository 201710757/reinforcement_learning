import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")

# LSTM Network Soon
class Mu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mu, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.mu = nn.Linear(64, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        m = torch.tanh(self.mu(x))
        return m




class Q(nn.Module):
    def __init__(self, input_dim):
        super(Q, self).__init__()

        self.input_dim = input_dim

        self.fc_s = nn.Linear(self.input_dim, 128)
        self.fc_a = nn.Linear(1, 128)
        self.fc_Q = nn.Linear(256, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, s, a):
        _s = F.relu(self.fc_s(s))
        _a = F.relu(self.fc_a(a))
        sa = torch.cat([_s, _a], dim=1)

        Q = F.relu(self.fc_Q(sa))
        Q = self.out(Q)

        return Q
