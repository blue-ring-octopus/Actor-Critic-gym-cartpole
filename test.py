# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:59:55 2022

@author: hibad
"""
import gym 
import numpy as np
from controller import Controller
from actor import ActorNet
from critic import CriticNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

critic = CriticNet()
critic.load_state_dict(torch.load("critic_param.p")['model_state_dict'])
critic.eval()
critic.to(device)

actor = ActorNet()
actor.load_state_dict(torch.load("actor_param.p")['model_state_dict'])
actor.eval()
actor.to(device)

player=Controller(critic, actor)



score=[]
for i in range(10):
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    terminated=False
    t=0
    while not terminated:
         action = player.input_((state,terminated))
         state_next, reward, terminated, truncated , info = env.step(action)  
         state = state_next
         
         env.render()
         t+=1
    score.append(t)
    env.close()

print(np.mean(score))