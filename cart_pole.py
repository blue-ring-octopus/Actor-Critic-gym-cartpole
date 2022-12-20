# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 02:20:36 2022

@author: hibad
"""
import gym 
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from torch import optim
# from copy import deepcopy
import matplotlib.pyplot as plt

from controller import Controller
from actor import ActorNet
from critic import CriticNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
critic_param=torch.load("critic_param.p")
episode=critic_param["episode"]
critic_losses=critic_param["loss"]

critic = CriticNet()
critic.load_state_dict(critic_param["model_state_dict"])
critic.eval()
critic.to(device)

actor_param=torch.load("actor_param.p")
actor_losses=actor_param["loss"]

actor = ActorNet()
actor.load_state_dict(actor_param["model_state_dict"])
actor.eval()
actor.to(device)

env = gym.make("CartPole-v1", render_mode="human")

player=Controller(critic, actor)
if critic_param["optimizer_state_dict"]:
    player.critic_optimizer.load_state_dict(critic_param["optimizer_state_dict"])

if actor_param["optimizer_state_dict"]:
    player.actor_optimizer.load_state_dict(actor_param["optimizer_state_dict"])

#%%
time=torch.load("training_scores.p")


learned=False
render=False

for i in range(5):
    if learned:
        render=True
        learned=False
        
    state, _ = env.reset()
    terminated=False
    t=0
    while not terminated:
         action = player.input_((state,terminated))
         state_next, reward, terminated, truncated , info = env.step(action)
         # if terminated:
         #     reward=-10
         loss=player.collect((state_next,terminated), reward,terminated)
         if loss[0]:
             learned=True
             critic_losses.append(loss[0])
             actor_losses.append(loss[1])
             
             print(loss)
             plt.figure()
             plt.plot(range(len(critic_losses)), critic_losses, "r.", alpha=0.1)
             plt.title("Critic Loss")
             plt.xlabel("Batch")
             plt.ylabel("MSE Loss")
             
             plt.figure()
             plt.plot(range(len(actor_losses)), actor_losses, "b.", alpha=0.1)
             plt.title("Actor Loss")
             plt.xlabel("Batch")
             plt.ylabel("Loss")
             plt.pause(0.05)
         state = state_next
         
         if render:
             env.render()
         t+=1
    if learned:
        time.append(t)
        plt.figure()
        plt.plot(range(len(time)), time, "k.")
        plt.title("Survived time")
        plt.xlabel("Episode")
        plt.ylabel("time")
        plt.pause(0.05) 
env.close()
     
#%% 
torch.save({
            'episode': episode,
            'model_state_dict': actor.state_dict(),
            'optimizer_state_dict': player.actor_optimizer.state_dict(),
            'loss': actor_losses,
            },"actor_param.p")

torch.save({
            'episode': episode,
            'model_state_dict': critic.state_dict(),
            'optimizer_state_dict': player.critic_optimizer.state_dict(),
            'loss': critic_losses,
            },"critic_param.p")

torch.save(time, "training_scores.p")

plt.figure()
plt.plot(range(len(critic_losses)), critic_losses, "r.", alpha=0.1)
plt.title("Critic Loss")
plt.xlabel("Batch")
plt.ylabel("MSE Loss")
plt.savefig('Critic_Loss.png', dpi=300)

plt.figure()
plt.plot(range(len(actor_losses)), actor_losses, "b.", alpha=0.1)
plt.title("Actor Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.savefig('Actor_Loss.png', dpi=300)

plt.figure()
plt.plot(range(len(time)), time, "k.", alpha=0.1)
plt.title("Survived time")
plt.xlabel("Episode")
plt.ylabel("time")
plt.savefig('learning_curve.png', dpi=300)
