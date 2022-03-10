import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()

        self.affine = nn.LSTM(input_dim+2, hidden_dim)
        
        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        if len(self.actions) > 0 and len(self.rewards) > 0:
            state = torch.cat([state.squeeze(0).squeeze(0), torch.FloatTensor([self.rewards[-1]]).to(device), torch.FloatTensor([self.actions[-1]]).to(device)]).unsqueeze(0)
        else:
            state = torch.cat([state.squeeze(0).squeeze(0), torch.FloatTensor([0.0]).to(device), torch.FloatTensor([0.0]).to(device)]).unsqueeze(0)
        """
        state = torch.tensor(state).unsqueeze(0)
        state, _ = self.affine(state)

        state_value = self.value_layer(state)
        
        action_value = self.action_layer(state)

        return state_value, action_value
    
    def reset_lstm(self):
        self.affine.weight_hh_l0.data.fill_(0)
        #torch.nn.init.normal_(self.affine.weight)
