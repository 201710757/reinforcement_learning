import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class qnet(nn.Module):
    def __init__(self, inputs, outputs):
        super(qnet, self).__init__()
        self.layer1 = nn.Linear(inputs, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 32)
        self.head = nn.Linear(32, outputs)
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.head(x)
