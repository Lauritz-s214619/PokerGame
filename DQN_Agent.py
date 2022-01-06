import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##DQN

#Det neurale netværk som approksimere q-værdier
class DQN(nn.Module):
    def __init__(self,hand,wallet,bet,community_cards):
        
        #Basic totalforbundet netværk (Hvad)
        self.fc1 = nn.Linear(in_features=hand * community_cards - \
                             bet + wallet, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=4)
    

    # t bliver kørt igennem netværket, med ReLU aktiveringsfunktion
    def forward(self,t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


##Experience

Experience = namedtuple('Experience',('state','action','next_state','reward'))

class ReplayMemory():
    #Memory er der vi gemmer experience
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    #Tilføjer Memory og sletter gammel ved maks kapacitet.
    def push(self,experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1   
    
    #Vælger tilfældig sample fra memory 
    def sample(self,batch_size):
         return random.sample(self.memory, batch_size)
    
    #Vi tjekker om der er nok memory til at vælge en experience
    def can_provide_sample(self, batch_size):
         return len(self.memory) >= batch_size
        
##Epsilon Greedy Strategy

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end 
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start-self.end) * math.exp(-1. * \
                           current_step * self.decay)

##Reinforcement Learning Agent

class Agent():
    #Skal have en epsilon Greedy strategy og antal actions
    def __init__(self,strategy, num_actions):
        self.current_step=0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self,state,policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1








