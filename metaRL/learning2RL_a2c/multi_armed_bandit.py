import random
import torch
class MAB:
    def __init__(self, n=2):
        #self.prob = [0.1, 0.9]
       
        #self.prob = self.prob[::-1] if random.random() < 0.5 else self.prob
        
        #m = torch.nn.Softmax(dim=0)
        #r = random.random()
        #self.prob = m(torch.tensor([r, 1-r])).numpy()
        
        r = round(torch.empty(1).uniform_(0,1).item(), 2)
        self.prob = [round(r, 2), round(1-r, 2)]


    def pull(self, action):
        if random.random() < self.prob[action]:
            reward = 1.0
        else:
            reward = 0.0
        return reward
