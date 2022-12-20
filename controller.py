# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:00:35 2022

@author: hibad
"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Controller:
    def __init__(self, critic_net, actor_net, gamma=0.99):
        self.stm=[]
        self.ltm=[]
        self.epsiode=[]
        
        self.critic_net=critic_net
        self.critic_optimizer=optim.Adam(critic_net.parameters())
        self.criterion=nn.MSELoss()
        
        self.actor_net=actor_net
        self.actor_optimizer=optim.Adam(actor_net.parameters())

        self.gamma=0.99

    def train_critic(self):
        lambda_=0.9
        critic_losses=[]
        for _ in range(5):
            print("collecting batch")

            batch=deepcopy(self.stm)
            if len(self.ltm)>5:
                for _ in range(5):
                    batch+=[deepcopy(self.ltm[np.random.randint(0,len(self.ltm))])]
                    
            target=[]
            inputs=[]
            print("process episode")
            for episode in batch:
                T=len(episode)
                V_t=np.zeros(T)
                r_t=np.zeros(T)
                for i,(s,a,r,s_prime) in enumerate(episode):
                    if s_prime[1]:
                        Q_prime=-10
                    else:
                        Q_prime=self.critic_net(torch.from_numpy(s_prime[0].astype(np.float32)).to(device))
                        Q_prime=Q_prime.cpu().detach().numpy()
                        
                    V_t[i]=r+self.gamma*Q_prime
                    r_t[i]=r
                    
                    Q=self.critic_net(torch.from_numpy(s[0].astype(np.float32)).to(device))
                    inputs.append(Q)

                G=[np.array([V_t[-1]])]
                
                for t in reversed(range(T-1)):
                    G_n=np.zeros(T-t)
                    G_n[0]=r_t[t]+self.gamma*V_t[t+1].copy()
                    G_n[1:]=r_t[t]+self.gamma*G[-1].copy()
                    
                    G.append(np.array(G_n))  

                for t in range(len(G)):
                    target.append(sum([lambda_**(n+1)*G[-t-1][n] if n==T-t-1 else (1-lambda_)*lambda_**(n)*G[-t-1][n] for n in range(T-t)]))

            print("gradient descent")
            target=torch.from_numpy(np.asarray(target).astype(np.float32)).to(device)
            inputs=torch.cat(inputs, dim=0)      
            inputs=inputs.unsqueeze(0)
            target=target.unsqueeze(0)
            
            self.critic_optimizer.zero_grad()

            critic_loss=self.criterion(inputs,target)

            critic_loss.backward()

            self.critic_optimizer.step()

            critic_losses.append(critic_loss.item())
        return critic_losses
    
    def train_actor(self):
        actor_losses=[]
        batch=[]
        for episode in deepcopy(self.stm):
            batch+=episode
            
        if len(self.ltm)>5:
            for _ in range(5):
                batch+=deepcopy(self.ltm[np.random.randint(0,len(self.ltm))])
        
        advantage=[]
        log_probs=[]
        for s,a,r,s_prime in batch:
            _, log_prob=self.policy(s[0])
            
            baseline=self.critic_net(torch.from_numpy(s[0]).to(device))
           
            advantage.append(r+self.gamma*self.critic_net(torch.from_numpy(s_prime[0]).to(device))-baseline)
            log_probs.append(log_prob(torch.from_numpy(np.array([a])).to(device)))

        advantage=torch.cat(advantage, dim=0).detach()
        log_probs = torch.cat(log_probs, dim=0)

        self.actor_optimizer.zero_grad()

        actor_loss = -(log_probs * advantage.detach()).sum()

        actor_loss.backward()

        self.actor_optimizer.step()

        actor_losses.append(actor_loss.item())
        
        return np.array(actor_losses)/len(batch)
    
    def train(self):
        print("learning")
        critic_losses=self.train_critic()
        actor_losses=self.train_actor()

       
        return np.mean(critic_losses), np.mean(actor_losses)
        
        
    def collect(self,s_prime, reward, terminated):
        loss=(0,0)
        self.epsiode.append(deepcopy((self.state, self.a, reward,s_prime)))
        
        if terminated:
            self.stm.append(self.epsiode)
            self.epsiode=[]
            
        if len(self.stm)>=5:
                loss=self.train()
                self.ltm+=deepcopy(self.stm)
                self.ltm=self.ltm[-5000:]
                self.stm=[]        
        return loss
    
    def policy(self,state):
        vec=torch.from_numpy(state.astype(np.float32)).to(device)
        distribution = self.actor_net(vec)
        action = distribution.sample()
        action_idx=action.cpu().numpy()
        return action_idx, distribution.log_prob
        
    def input_(self, state):
        self.state=deepcopy(state)
        a,_=self.policy(state[0])
        self.a=deepcopy(a)
        
        return a        