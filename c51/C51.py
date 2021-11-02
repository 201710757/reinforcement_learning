import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from DQN import qnet
from ReplayMemory import ReplayMemory
#import gym

#env = gym.make('CartPole-v1')
#ACTION_SPACE = env.action_space.n
#OBSERVATION_SPACE = env.observation_space.shape[0]

REPLAY_MEMORY = 10000
BATCH_SIZE = 128
LR = 1e-4
GAMMA = 0.99

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N_ATOM = 51
V_MIN = -10.
V_MAX = 10.
V_STEP = ((V_MAX-V_MIN)/(N_ATOM-1))
V_RANGE = np.linspace(V_MIN, V_MAX, 51) # this is why C51

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

TARGET_UPDATE_FREQUENCY = 20

class C51(nn.Module):
    def __init__(self, inputs, outputs, n_action):
        super(C51, self).__init__()
        self.N_ACTIONS = n_action
        self.N_SPACE = inputs

        self.pred_net, self.target_net = qnet(inputs, outputs), qnet(inputs, outputs)
        self.update_target()
        
        self.pred_net.to(device)
        self.target_net.to(device)

        self.memory_counter = 0
        self.learn_step_counter = 0

        self.replay_buffer = ReplayMemory(REPLAY_MEMORY)

        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)
        
        self.value_range = torch.FloatTensor(V_RANGE)
        self.value_range.to(device)
    
    def pick_action(self, x, ep):
        x = torch.FloatTensor(x)
        x = x.to(device)
        
        if np.random.uniform() > ep:
            action_value_dist = self.pred_net(x)
            action_value = torch.sum(action_value_dist * self.value_range.view(1,1,-1), dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            action = np.random.randint(0, self.N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, n_s, done):
        self.memory_counter += 1
        self.replay_buffer.push(s,a,r,n_s,done)

    def learn(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target()

        # b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(BATCH_SIZE)
        minibatch = self.replay_buffer.sample(BATCH_SIZE)
        b_s = [x[0] for x in minibatch]
        b_a = [x[1] for x in minibatch]
        b_r = [x[2] for x in minibatch]
        b_ns = [x[3] for x in minibatch]
        b_d = [x[4] for x in minibatch]

        b_w, b_idxes = np.ones_like(b_r), None

        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.FloatTensor(b_a).to(device)
        b_ns = torch.FloatTensor(b_ns).to(device)

        q_eval = self.pred_net(b_s)
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1)

        q_target = np.zeros((mb_size, 51))
        
        q_next = self.target_net(b_ns).detach()
        q_next_mean = torch.sum(q_next * self.value_range.view(1,1,-1), dim=2)
        actions = q_next_mean.argmax(dim=1)
        q_next = torch.stack([q_next[i].index_select(0, actions[i]) for i in range(mb_size)]).squeeze(1)
        q_next = q_next.data.cpu().numpy()

        # projection
        next_v_range = np.expand_dims(b_r, 1) + GAMMA * np.expand_dims((1.0 - b_d), 1) * np.expand_dims(self.value_range.data.cpu().numpy(), 0)
        next_v_pos = np.zero_like(next_v_range)
        next_v_range = np.clip(next_v_range, V_MIN, V_MAX)
        next_v_pos = (next_v_range - V_MIN) / V_STEP
        
        lb = np.floor(next_v_pos).astype(int)
        ub = np.cell(next_v_pos).astype(int)

        for i in range(mb_size):
            for j in range(N_ATOM):
                q_target[i, lb[i, j]] += (q_next * (ub - next_v_pos))[i, j]
                q_target[i, ub[i, j]] += (q_next * (next_v_pos - lb))[i, j]
        q_target = torch.FloatTensor(q_target)
        q_target = q_target.to(device)

        loss = q_target * (-torch.log(q_eval + 1e-8))
        loss = torch.mean(loss)
        
        b_w = torch.Tensor(b_w)
        b_w = b_w.to(device)

        loss = torch.mean(b_w*loss)
        
        if self.learn_step_counter % 100 == 0 and self.learn_step_counter > 1: 
            print("Loss : ", loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.pred_net.state_dict())








