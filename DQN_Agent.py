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
    

## Parametre

batch_size = 256
gamma = 0.999 
eps_start = 1  
eps_end = 0.01      
eps_decay = 0.001
target_update = 10
memory_size = 10000 
lr = 0.001
num_episodes = 1000000

##Main Program

device = troch.device("cuda" if torch.cuda.is_available() else "cpu")
#em
strategy = EpsilonGreedyStrategy(eps_start,eps_end, eps_decay)
agent = Agent(strategy, #em
memory = ReplayMemory(memory_size)

policy_net = DQN(#em)
target_net = DQN(#em)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizier = optim.Adam(params=policy_net.parameters(), lr=lr)

for episode in range(num_episodes):
    #em.reset(
    state = #VI skal have state her
    
    for timestep in count():
        action = agent.select_action(state,policy_net)
        reward = #Hvad er reward?
        next_state = #Hvor ender vi efter?
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            
            current_q_values = QValues.get_current(policy_net,states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards 
            
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if #em.done:
            #plot?
            break
        
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        

## Tensor processing

def extract_tensort(experiences):
    
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch_reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1,t2,t3,t4)

## Q-Value Calculator
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detacht()
        return values
        
        
## DQN
    
# Det neurale netværk som approksimere q-værdier
class DQN(nn.Module):
    def __init__(self,hand,wallet,actions,community_cards,number_of_players,Highest_raise,small_blind):
        
        #Basic totalforbundet netværk (Hvad)
        self.fc1 = nn.Linear(in_features=hand, community_cards, \
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


## Experience

Experience = namedtuple('Experience',('state','action','next_state','reward'))

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
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end 
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start-self.end) * math.exp(-1. * \
                           current_step * self.decay)

## Reinforcement Learning Agent

class Agent():
    #Skal have en epsilon Greedy strategy og antal actions (og gpu/cpu)
    def __init__(self,strategy, num_actions, device):
        self.current_step=0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self,state,policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device)   #  explore
        else:
            # no_grad da vi ikke vil opdatere gradienten til NN
            with torch.no_grad():
                return policy_net(state.argmax(dim=1).item().to(device) # exploit
                              
                        






