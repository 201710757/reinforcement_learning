import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()

        # self.affine = nn.Linear(input_dim, hidden_dim)
        # self.state1 = nn.Linear(hidden_dim, hidden_dim)
        # self.value1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.memory = nn.LSTM(input_dim, hidden_dim)

        self.h0 = nn.Parameter(torch.randn(1, 1, self.memory.hidden_size).float())
        self.c0 = nn.Parameter(torch.randn(1, 1, self.memory.hidden_size).float())

        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        #state = self.affine(state)
        
        #state1 = self.state1(F.relu(state))
        state, mem = x
        state, mem = self.memory(state.unsqueeze(0), mem)
        state_value = self.value_layer(F.relu(state))

        
        #value = self.value1(F.relu(state))
        action_value = self.action_layer(F.relu(state))

        return state_value, action_value, mem
    
    def get_init_states(self):
        return (self.h0, self.c0)
