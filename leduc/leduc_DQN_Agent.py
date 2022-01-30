import math
import random
from sched import scheduler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from leduc import *
import os

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

## Q-Value Calculator
class QValues():
    print(device)
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states, valid_actions):
        return torch.tensor([float(max(el[valid_actions[idx]])) for idx,el in enumerate(target_net(next_states))]).float()
        
        
## DQN
    
# Det neurale netværk som approksimere q-værdier
class DQN(nn.Module):
    def __init__(self):
        
        #Basic totalforbundet netværk (Hvad)
        #Input features = 2 hand, 5 community, wallet og bet. 
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=32).to(device)
        #self.fc2 = nn.Linear(in_features=32, out_features=32).to(device)
        self.out = nn.Linear(in_features=32, out_features=4).to(device)
    
    
    # t bliver kørt igennem netværket, med ReLU aktiveringsfunktion
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        #t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


## Experience

Experience = namedtuple('Experience',('state','action','next_state','reward', 'valid_actions'))

class ReplayMemory():
    # Memory er der vi gemmer experience
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    # Tilføjer Memory og sletter gammel ved maks kapacitet.
    def push(self,experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1   
    
    # Vælger tilfældig sample fra memory 
    def sample(self,batch_size):
         return random.sample(self.memory, batch_size)
    
    # Vi tjekker om der er nok memory til at vælge en experience
    def can_provide_sample(self, batch_size):
         return len(self.memory) >= batch_size
        
## Epsilon Greedy Strategy

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay, steps):
        self.start = start
        self.end = end 
        self.decay = decay
        self.steps = steps

    def get_exploration_rate(self, current_step):
        return self.end + (self.start-self.end) * math.exp(-1. * current_step * self.decay)
        #return self.start - min(current_step/self.steps,1)*(self.start-self.end)

## Reinforcement Learning Agent

class Agent():
    #Skal have en epsilon Greedy strategy og antal actions (og gpu/cpu)
    def __init__(self,strategy, device):
        self.current_step=0
        self.strategy = strategy
        self.device = device
        self.eps = 0

    def select_action(self, state, policy_net, valid_actions):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.eps = rate
        self.current_step += 1

        if rate > random.random():
            return random.choice(valid_actions)   #  explore
        else:
            # no_grad da vi ikke vil opdatere gradienten til NN
            with torch.no_grad():
                out = policy_net(state)
                valid_out = out[:,np.array(valid_actions)]
                idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                return idx.item() # exploit



