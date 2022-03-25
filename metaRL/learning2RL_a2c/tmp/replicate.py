import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from two import two_step_task
#from C_MDP import CustomEnv

import time
import matplotlib.pyplot as plt
import numpy as np
import gym
from ActorCritic import ActorCritic
from multi_armed_bandit import MAB
import torch.multiprocessing as mp


import time
device = torch.device("cuda:0")
env_name = 'A2C_Two_Step_Task_' + time.ctime(time.time())
env = two_step_task()

writer = SummaryWriter("runs/"+ env_name)
ENV_RESET_TERM = 10

input_dim = env.nb_states
hidden_dim = 48
output_dim = env.num_actions
LR = 7e-4
MAX_EP = 3000
GAMMA = 0.9

def train():
    policy = ActorCritic(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr = LR)
    #env = two_step_task()
    
    train_reward = []
    for ep in range(MAX_EP):
 
        a = [0]*output_dim
        r = 0
        t = 0
        ep_reward = 0

        log_prob_actions = []
        state_values = []
        rewards = []

        p_action, p_reward = [0]*output_dim, 0

        step = 0
        
        #if ep % ENV_RESET_TERM == 0:
        rnn_state = policy.init_lstm_state()
            #env = two_step_task()#MAB(k)

            #prob_list.append(env.prob)

        s = env.reset()
        d = False

        if (env.state == env.S_1):
            env.possible_switch()
        while not d:
            step += 1
            #print(s, a, r, t)
            s = np.concatenate([s,[t]]) #s + a + [r] + [t]#[s, a, r, step/100.0]
            #print("STATE : ", s)
            s = torch.FloatTensor(s).to(device)#.unsqueeze(0)
            state_pred, action_pred, rnn_state = policy(
                    s,
                    (
                        torch.tensor(p_action).float().to(device), 
                        torch.tensor([p_reward]).float().to(device), 
                    ),
                    rnn_state
                )
            # rnn_state = rnn_state[0].detach(), rnn_state[1].detach() 
            # test version
            
            action_prob = F.softmax(action_pred, dim=-1)
            try:
                #print(action_prob)
                dist = Categorical(action_prob)
            except:
                print("---PRED---")
                print(action_pred)
                print("---PROB---")
                print(action_prob) 
            action = dist.sample()
            a = action.item()

            s1, r, d, t = env.step(a)

            p_action = np.eye(output_dim)[a]
            p_reward = r
            log_prob_actions.append(dist.log_prob(action))
            state_values.append(state_pred)
            rewards.append(r)

            ep_reward += r

            s = s1
        
        
        log_prob_actions = torch.cat(log_prob_actions).to(device)
        state_values = torch.cat(state_values).squeeze(-1).to(device)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA*R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(device)
        #returns = (returns - returns.mean()) / (returns.std()+)
        
        advantage = returns - state_values
        #advantage = (advantage - advantage.mean()) / advantage.std()

        advantage = advantage.detach()
        returns = returns.detach()
        
        # actor loss
        action_loss = -(advantage * log_prob_actions).sum()
        
        # critic loss
        value_loss = F.mse_loss(state_values, returns).sum()
        
        loss = action_loss + value_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if ep % int(ENV_RESET_TERM/10) == 0 and ep != 0:
            print("ep {} - reward {}".format(ep, np.mean(train_reward)))
            writer.add_scalar(str(ENV_RESET_TERM/10) +" ep mean reward", np.mean(train_reward), ep)
            train_reward = []

        train_reward.append(ep_reward)
    plot(ep-1)
def plot(episode_count):
    fig, ax = plt.subplots()
    x = np.arange(2)
    ax.set_ylim([0, 1.2])
    ax.set_ylabel('Stay Probability')

    stay_probs = env.stayProb()

    common = [stay_probs[0,0,0], stay_probs[1,0,0]]
    uncommon = [stay_probs[0,1,0], stay_probs[1,1,0]]

    ax.set_xticks([1.3,3.3])
    ax.set_xticklabels(['Last trial rewarded', 'Last trial not rewarded'])

    c = plt.bar([1,3], common, color='b', width=0.5)
    uc = plt.bar([1.8,3.8], uncommon, color='r', width=0.5)
    ax.legend((c[0], uc[0]), ('common', 'uncommon'))
    plt.savefig("TEST_" + str(episode_count) + '.png')
    env.transition_count = np.zeros((2,2,2))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


train()
#df = pd.DataFrame(np.array(prob_list))
#df.to_csv('prob_list.csv')
