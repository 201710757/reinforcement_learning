import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        x = x.to(device)
        x = self.layer(x)
        return x

model = QNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)

loss_func = nn.MSELoss()

# s = env.reset()
# Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 10000
rList = []

g = 0.98

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    e = max(0.01, 0.08 - 0.01*(i/200))
    
    while not done:
        # greedy act
        state = torch.tensor(state)
        x_qs = Variable(state).to(device)
        Qs = model.forward(x_qs.float())
        # print("QS : {}".format(Qs))
        
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(Qs.argmax())
        #print(action)
        new_state, reward, done, info = env.step(action)
        new_state = torch.tensor(new_state)
        if done:
            Qs[action] = -100
        else:
            x_qs1 = Variable(new_state).to(device)
            Qs1 = model.forward(x_qs1.float())
            #print("QS1 {}".format(Qs1.argmax()))
            Qs[action] = reward + g*Qs1.max()
            # print(g*Qs1.max())
        
        # Update Q value
        loss = loss_func(state, Qs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = new_state
        rAll += reward
    rList.append(rAll)

print("Success rate : ", str(sum(rList)/num_episodes))

import matplotlib.pyplot as plt
plt.bar(range(len(rList)), rList, color='blue')
plt.show()
