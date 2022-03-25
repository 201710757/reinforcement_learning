import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()
        
        #self.encoder = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim+1+2, hidden_dim)

        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.action_layer.weight.data, 0.01)
        self.action_layer.bias.data.fill_(0)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.value_layer.weight.data, 1)
        self.value_layer.bias.data.fill_(0)

    def forward(self, state, past_in, memory=None):

        if memory == None:
            memory = self.init_lstm_state()
        #state = self.encoder(state)
        
        state = torch.cat((state, *past_in), dim=-1).unsqueeze(0)
        state, memory = self.lstm(state.unsqueeze(0), memory)
        

        state_value = self.value_layer(state)
        
        action_value = self.action_layer(state)

        return state_value, action_value, memory
    
    def init_lstm_state(self):
        h0 = torch.zeros(1, 1, self.lstm.hidden_size).float().to(device)
        c0 = torch.zeros(1, 1, self.lstm.hidden_size).float().to(device)
        return (h0, c0)
