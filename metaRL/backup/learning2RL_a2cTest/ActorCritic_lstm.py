import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(input_dim, 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
        )
        self.hidden_dim = hidden_dim
        
        self.affine = nn.Linear(input_dim, hidden_dim)
        #self.affine1 = nn.Linear(hidden_dim, hidden_dim)
        self.meta_LSTM = nn.LSTM(hidden_dim, hidden_dim)
        
        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, mem_state):
        """
        if len(self.actions) > 0 and len(self.rewards) > 0:
            state = torch.cat([state.squeeze(0).squeeze(0), torch.FloatTensor([self.rewards[-1]]).to(device), torch.FloatTensor([self.actions[-1]]).to(device)]).unsqueeze(0)
        else:
            state = torch.cat([state.squeeze(0).squeeze(0), torch.FloatTensor([0.0]).to(device), torch.FloatTensor([0.0]).to(device)]).unsqueeze(0)
        """
        state = torch.tensor(state).unsqueeze(0)
        # self.memory = (self.memory[0].to(device), self.memory[1].to(device))
        state = self.affine(state)
        state, memory = self.meta_LSTM(state, mem_state) 
        #state = self.affine1(state)
        state_value = self.value_layer(state)
        
        action_value = self.action_layer(state)

        return state_value, action_value, memory
    
    def reset_lstm(self):
        memory = (torch.zeros(1,1,self.hidden_dim).float().to(device), torch.zeros(1,1,self.hidden_dim).float().to(device))
        #self.affine.weight_hh_l0.data.fill_(0)
        return memory
