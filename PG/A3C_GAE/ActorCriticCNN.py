import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, 4, stride=2, padding=1),#, kernel_size=(8,8), stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),#, kernel_size=(4,4), stride=(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),# kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, hidden_dim)

            #nn.ReLU()
        )
        self.affine = nn.Linear(hidden_dim, hidden_dim)

        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        state = self.encoder(state)

        state = self.affine(F.relu(state))
        state_value = self.value_layer(state)

        action_value = self.action_layer(state)

        return state_value, action_value
