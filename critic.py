# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:04:04 2022

@author: hibad
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(4,100)
        torch.nn.init.xavier_uniform_(self.fc1 .weight)

        self.fc2 = nn.Linear(100,100)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(100,1)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x):
        x =F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        x=torch.clip(x, min=-10, max=100)
        return x
    
if __name__ == "__main__":
    model=CriticNet()    
    torch.save({
                'episode': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,
                'loss': [],
                },"actor_param.p")
    

