import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_Actor = nn.Linear(hidden_dim, output_dim)
        self.fc_Critic = nn.Linear(hidden_dim, 1)

    def actor(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x = self.fc_Actor(x)
        prob = F.softmax(x, dim = softmax_dim)

        return prob

    def critic(self, x):
        x = F.relu(self.fc1(x))
        cri = self.fc_Critic(x)
        
        return cri
