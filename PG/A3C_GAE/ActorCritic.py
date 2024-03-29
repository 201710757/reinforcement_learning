import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()


        self._affine = nn.Linear(input_dim, hidden_dim)
        self.affine = nn.Linear(hidden_dim, hidden_dim)

        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        state = self._affine(state)
        state = F.relu(state)
        state = self.affine(state)
        state = F.relu(state)


        state_value = self.value_layer(state)

        action_value = self.action_layer(state)

        return state_value, action_value
