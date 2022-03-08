import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(Critic, self).__init__()

        self.affine = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.affine(x)
        x = F.relu(x)
        x = self.value_layer(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(Actor, self).__init__()

        self.affine = nn.Linear(input_dim, hidden_dim)

        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.affine(x)
        x = F.relu(x)
        x = F.softmax(self.action_layer(x), dim=0)

        return x

    def loss(self, GAMMA=0.99, normalize = True):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        # print(returns)
        returns = torch.tensor(returns).to(device)
        if normalize:
            returns = (returns - returns.mean()) / returns.std()

        # will be reduced - code
        loss = 0
        for logprob, value, reward in zip(self.log_prob_actions, self.state_values, returns):
            advantage = reward - value.reshape(-1)
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)

            loss += (action_loss.sum().reshape(-1).float() + value_loss.float())
            #print(loss)
        return loss

    # Why?
    def clearMemory(self):
        del self.log_prob_actions[:]
        del self.state_values[:]
        del self.rewards[:]
