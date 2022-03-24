import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")

class Mu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mu, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 1024)
        # = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        # = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 64)
        # = nn.BatchNorm1d(64)
        self.mu = nn.Linear(64, self.output_dim)

    def forward(self, x):
        #x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        m = torch.tanh(self.mu(x))
        
        return m




class Q(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc_s = nn.Linear(self.input_dim, 1024)
        # = nn.BatchNorm1d(512)
        
        self.fc_a = nn.Linear(self.output_dim, 256)
        # = nn.BatchNorm1d(256)
        self.fc_a1 = nn.Linear(256,1024)
        # = nn.BatchNorm1d(512)

        self.fc_Q = nn.Linear(2048, 2048)
         #= nn.BatchNorm1d(1024)
        self.fc_Q1 = nn.Linear(2048, 128)
         #= nn.BatchNorm1d(64)

        self.out = nn.Linear(128, 1)

    def forward(self, s, a):
        #s = s.unsqueeze(0)
        #a = a.unsqueeze(0)

        _s = F.relu(self.fc_s(s))
        _a = F.relu(self.fc_a(a))
        _a = F.relu(self.fc_a1(_a))

        sa = torch.cat([_s, _a], dim=1)

        Q = F.relu(self.fc_Q(sa))
        Q = F.relu(self.fc_Q1(Q))
        Q = self.out(Q)

        return Q
