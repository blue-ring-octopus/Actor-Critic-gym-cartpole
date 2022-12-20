# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:53:20 2022

@author: hibad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(4,100)
        torch.nn.init.xavier_uniform_(self.fc1 .weight)

        self.fc2 = nn.Linear(100,100)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(100,2)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x):
        x =F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution
    
    
if __name__ == "__main__":
    model=ActorNet()    
    torch.save({
                'episode': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,
                'loss': [],
                },"actor_param.p")
    
