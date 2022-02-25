import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(ActorCritic, self).__init__()

        self.affine = nn.Linear(input_dim, hidden_dim)

        # Actor
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.value_layer = nn.Linear(hidden_dim, 1)

        # dont use it now..
        self.dropout = nn.Dropout(dropout)
        
        self.log_prob_actions = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        # state = torch.from_numpy(state).float()
        state = self.affine(state)
        state = F.relu(state)
        state_value = self.value_layer(state)
        # state_value = self.dropout(state_value)

        action_prob = F.softmax(self.action_layer(state), dim=1)
        dist = Categorical(action_prob)
        action = dist.sample()

        self.log_prob_actions.append(dist.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def loss(self, GAMMA=0.99, normalize = True):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        if normalize:
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / returns.std()

        print(self.state_values)
        print("---------")
        prnit(returns)
        # will be reduced - code
        loss = 0
        for logprob, value, reward in zip(self.log_prob_actions, self.state_values, returns):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)

            loss += (action_loss + value_loss)
        return loss

    # Why?
    def clearMemory(self):
        del self.log_prob_actions[:]
        del self.state_values[:]
        del self.rewards[:]
