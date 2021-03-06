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
        self.actions = []

    def forward(self, state):
        #state = state.to(device)
        if len(self.actions) > 0 and len(self.rewards) > 0:
            state = torch.cat([state.squeeze(0).squeeze(0), torch.FloatTensor([self.rewards[-1]]).to(device), torch.FloatTensor([self.actions[-1]]).to(device)]).unsqueeze(0)
        else:
            state = torch.cat([state.squeeze(0).squeeze(0), torch.FloatTensor([0.0]).to(device), torch.FloatTensor([0.0]).to(device)]).unsqueeze(0)
        state = torch.tensor(state).unsqueeze(0)
        state, _ = self.affine(state)

        state_value = self.value_layer(state)
        

        
        action_prob = F.softmax(self.action_layer(state), dim=-1)
        dist = Categorical(action_prob)
        action = dist.sample()
        self.actions.append(action)

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
        del self.actions[:]
