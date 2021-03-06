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
        self.actor_aff = nn.Linear(hidden_dim, hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, output_dim)

        # Critic
        self.critic_aff = nn.Linear(hidden_dim ,hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

        # use it now!
        self.dropout = nn.Dropout(dropout)
        
        self.log_prob_actions = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        # state = torch.from_numpy(state).float()
        state = self.affine(state)
        state = F.relu(state)
        state = self.dropout(state)

        state_v = self.critic_aff(state)
        state_v = F.relu(state_v)
        state_v = self.dropout(state_v)
        state_value = self.value_layer(state_v)
        # state_value = self.dropout(state_value)
        

        action_v = self.actor_aff(state)
        action_v = F.relu(action_v)
        action_v = self.dropout(action_v)
        action_prob = F.softmax(self.action_layer(action_v), dim=-1)
        dist = Categorical(action_prob)
        action = dist.sample()

        self.log_prob_actions.append(dist.log_prob(action))
        self.state_values.append(state_value)

        return action

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
