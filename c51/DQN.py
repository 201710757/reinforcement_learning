import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class qnet(nn.Module):
    def __init__(self, outputs):
        super(qnet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)
        
        self.fc_q = nn.Linear(512, outputs * 51)

    def forward(self, x):
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0)
        x = F.relu(self.fc(x))
        
        action_value = F.softmax(self.fc_q(x).view(mb_size, outputs, 51), dim=2)

        return action_value