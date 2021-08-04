import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class qnet(nn.Module):
    def __init__(self, inputs, outputs):
        super(qnet, self).__init__()
        self.layer1 = nn.Linear(inputs, outputs)
    def forward(self, x):
        x = x.to(device)
        x = self.layer1(x)
        return x
