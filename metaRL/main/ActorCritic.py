import numpy as np
import torch
import torch.nn as nn
import torhc.nn.functional as F
from torch.autograd import Variable

def ortho_init(weight, scale=1.):
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
        self.actor.weight.data = ortho_init(self.actor.weight_data, 0.01)
        self.actor.bias.data.fill_(0)
        
        self.critic = nn.Linear(hidden_dim, 1))
        self.critic.weight.data = orth_init(self.critic.weight_data, 1.0)
        self.critic.bias.data.fill_(0)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim)


        self.h0 = nn.Parameter(torch.randn(1,1, self.lstm.hidden_size).float())
        self.c0 = nn.Parameter(torch.randn(1,1, self.lstm.hidden_size).float())

    def forward(self, x):
        s, p_action, p_reward, t, mem_state = x
        p_input = torch.cat((s, p_action, p_reward, t), dim=-1)

        if mem_state is None:
            mem_state = (self.h0, self.c0)

        h_t, mem_state = self.lstm(p_input.unsqueeze(1), mem_state)

        action_dist = F.softmax(self.actor(h_t), dim=-1)
        value = self.critic(h_t)

        return action_dist, value, mem_state

    def get_init_states(self):
        return (self.h0, self.c0)
