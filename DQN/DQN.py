import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(inputs, 128, bias=True)
        self.layer2 = nn.Linear(128, 256, bias=True)
        self.layer3 = nn.Linear(256, 64, bias=True)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(64)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # def conv2d_size_out(size, kernel_size=5, stride=2):
        #     return (size-(kernel_size-1)-1) // stride + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32

        self.head = nn.Linear(64, outputs, bias=True)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        return self.head(x)

    def predict(self, state):
        #state = torch.tensor(state, dtype=torch.double)
        # state = torch.from_numpy(state, dtype=torch.double)
        state = torch.tensor(state).float()
        # state = torch.tensor(state, dtype=torch.float32)
        return self.forward(state).argmax()
