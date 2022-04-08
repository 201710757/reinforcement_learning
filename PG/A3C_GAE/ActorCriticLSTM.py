import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()

        self.memory = nn.LSTM(input_dim, hidden_dim)
        
        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)
        

        self.h0 = nn.Parameter(torch.randn(1,1, self.memory.hidden_size).float())
        self.c0 = nn.Parameter(torch.randn(1,1, self.memory.hidden_size).float())
        
    def forward(self, x):
        state, mem_state = x

        state, mem_state = self.memory(state.unsqueeze(1), mem_state)
        state = F.relu(state)
        state_value = self.value_layer(state)

        action_value = self.action_layer(state)

        return state_value, action_value, mem_state

    def get_init_states(self):
        return (self.h0, self.c0)
