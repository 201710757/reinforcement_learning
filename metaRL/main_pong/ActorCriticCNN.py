import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def ortho_init(weights, scale=1.):
    shape = tuple(weights.size())
    flat_shape = shape[1], shape[0]

    a = torch.tensor(np.random.normal(0., 1., flat_shape))

    u, _, v = torch.svd(a)
    t = u if u.shape == flat_shape else v
    t = t.transpose(1,0).reshape(shape).float()

    return scale * t

class A2C_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(A2C_LSTM, self).__init__()
        
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.actor.weight.data = ortho_init(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        self._actor = nn.Linear(hidden_dim, hidden_dim)

        
        self.critic = nn.Linear(hidden_dim, 1)
        self.critic.weight.data = ortho_init(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)
        self._critic = nn.Linear(hidden_dim, hidden_dim)
        

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8,8), stride=(4,4)),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=(3,3)),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1)),
            nn.Flatten(),
            nn.Linear(1024, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim + 8, hidden_dim)


        self.h0 = nn.Parameter(torch.randn(1,1, self.lstm.hidden_size).float())
        self.c0 = nn.Parameter(torch.randn(1,1, self.lstm.hidden_size).float())

    def forward(self, x):
        s, p_action, p_reward, t, mem_state = x
        s = self.encoder(s)
        p_input = torch.cat((s, p_action, p_reward, t), dim=-1)

        if mem_state is None:
            mem_state = (self.h0, self.c0)

        h_t, mem_state = self.lstm(p_input.unsqueeze(1), mem_state)

        a_ht = self._actor(F.relu(h_t))
        action_dist = F.softmax(self.actor(F.relu(a_ht)), dim=-1)

        v_ht = self._critic(F.relu(h_t))
        value = self.critic(F.relu(v_ht))


        return action_dist, value, mem_state

    def get_init_states(self):
        return (self.h0, self.c0)
