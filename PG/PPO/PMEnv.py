import random
import numpy as np
import gym
from math import exp

from numpy.random import choice
from gym import spaces
import torch
# https://github.com/awjuliani/Meta-RL

ACTION_SPACE = 2
N_STATE = 21
N_CTXT = 4
N_STAGE = 2
OBS_SPACE = N_STATE#+N_CTXT
TRANSITION_PROBABILITY = [[0.5, 0.5], [0.9, 0.1]]
REWARD_LIST = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
COLOR_LIST = ['YELLOW','BLACK','RED','GREEN','RED','YELLOW','YELLOW','GREEN','BLACK','RED','BLUE','BLACK','YELLOW','RED','RED','BLUE']
REWARD_COLOR = [7,8,8,-1,8,-1,7,-1,7,6,6,-1,7,-1,-1,6]
R_COLOR = [7, 8, -1, 6]

REWARD = [20, 10, 0, 40]
COLOR = ['YELLOW', 'BLACK', 'RED', 'GREEN', 'BLUE']

class PMEnv(gym.Env):
    def __init__(self):
        super(PMEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Discrete(OBS_SPACE)
        self.trans_prob = TRANSITION_PROBABILITY[1]
        self.GOAL_CONDITION = True #REWARD[0]
        self.bucket = random.choices(R_COLOR, weights=[0.25,0,0.25,0.25])[0]
        self.observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
    
    def change_goalCondition(self):
        _tmp = self.bucket
        while True:
            if not self.GOAL_CONDITION:
                self.bucket = random.choices(R_COLOR, weights=[0.25,0.25,0.25,0.25])[0]
            else:
                self.bucket = random.choices(R_COLOR, weights=[0.25,0.,0.25,0.25])[0]
            if self.bucket != _tmp:
                break

        print("bucket changed : ", _tmp, " -> ", self.bucket)
        #self.GOAL_CONDITION = random.choices([False, True], weights=[0.5, 0.5])[0] #True # REWARD[random.randrange(0,len(REWARD))]
        #print("Goal Condition : ", self.GOAL_CONDITION)

    def step(self, action):
        action_q = action

        L_R = random.choices([action, 1-action], weights=self.trans_prob)[0]


        #observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
        
        self.pos = self.pos*4+1 + action*2 + L_R

        self.observation[self.pos] = 1
        
        if(self.stg < 2):
            self.stg += 1
            self.agent_a.append(action_q)
        
        if (self.stg >= 2):
            done = True
            #print("pos : ", self.pos-5)
            if self.GOAL_CONDITION: # specific GC
                if REWARD_COLOR[self.pos-5] == self.bucket: #random.choices(R_COLOR, weights=[0.25,0.25,0.25,0.25])[0]:
                    #COLOR_LIST[self.pos - 5] == random.choices(COLOR, weights=[0.2,0.2,0.2,0.2,0.2])[0]: 
                    #REWARD_LIST[self.pos - 5] == 20 and COLOR_LIST[self.pos - 5] == 'YELLOW':
                    reward = 1 #REWARD_LIST[self.pos - 5] / 40
                    #reward = 50 if reward == 0 else reward
                else:
                    reward = 0
            else: # flexible GC
                reward = REWARD_LIST[self.pos - 5] / 40
                #reward = 50 if reward == 0 else reward
            #self.stg = 0
            #self.pos = 0
        else:
            done = False
            reward = 0
        # reward = reward * likelihood_

        return self.observation, reward, done, {}
    
    def reset(self):
        self.agent_a = []
        self.cti = 0
        self.stg = 0

        self.observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
        # state = int(self.mat[self.cti, self.stg])
        self.pos = 0 #state-1
        self.observation[self.pos] = 1

        #cidx = (int(self.mat[self.cti, -1]) % 5) + 15
        #observation[cidx] = 1
        return self.observation

    def render(self):
        pass

