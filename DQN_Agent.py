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
from itertools import count
import leduc
    

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

## Q-Value Calculator
class QValues():
    print(device)
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach().to(device)
        return values
        
        
## DQN
    
# Det neurale netværk som approksimere q-værdier
class DQN(nn.Module):
    def __init__(self):
        
        #Basic totalforbundet netværk (Hvad)
        #Input features = 2 hand, 5 community, wallet og bet. 
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=32).to(device)
        self.fc2 = nn.Linear(in_features=32, out_features=32).to(device)
        self.out = nn.Linear(in_features=32, out_features=4).to(device)
    
    
    # t bliver kørt igennem netværket, med ReLU aktiveringsfunktion
    def forward(self, t):
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
        return self.end + (self.start-self.end) * math.exp(-1. * current_step * self.decay)

## Reinforcement Learning Agent

class Agent():
    #Skal have en epsilon Greedy strategy og antal actions (og gpu/cpu)
    def __init__(self,strategy, device):
        self.current_step=0
        self.strategy = strategy
        self.device = device
        self.eps = 0

    def select_action(self, state, policy_net, valid_actions):
        rate = strategy.get_exploration_rate(self.current_step)
        self.eps = rate
        self.current_step += 1

        if rate > random.random():
            action = random.choice(valid_actions)
            return torch.tensor([action]).to(self.device)   #  explore
        else:
            # no_grad da vi ikke vil opdatere gradienten til NN
            with torch.no_grad():
                out = policy_net(state)
                valid_out = out[:,np.array(valid_actions)]
                idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                return idx.to(self.device) # exploit
                              


## Plot function
def plot(values, period):
    plt.figure(2)
    plt.clf()
    plt.title(f'Training, Eps: {agent.eps}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    avg_reward = get_avg_reward(period, values)
    plt.plot(avg_reward)  
    plt.pause(0.001)
    #print("Episode", len(values), "\n", period, "episode moving avg:", avg_reward[-1])
    #if is_ipython: display.clear_output(wait=True)

def get_avg_reward(period, values):
    values = torch.tensor(values, dtype=torch.float).to(device)
    if len(values) >= period:
        avg_reward = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        avg_reward = torch.cat((torch.zeros(period-1).to(device), avg_reward))
        return avg_reward.cpu().numpy()
    else:
        avg_reward = torch.zeros(len(values))
        return avg_reward.numpy()                        

## Tensor processing

def extract_tensors(experiences):
    
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1,t2,t3,t4)


## Parametre

batch_size = 32
gamma = 0.99    
eps_start = 1  
eps_end = 0.1      
eps_decay = 0.0005
target_update = 1000
memory_size = 20000
lr = 0.00005
num_episodes = 10000

##Main Program

player = leduc.Player("Billy Ailish", 10)
algo = leduc.Player("Bob", 10)
player.role = leduc.Roles.SmallBlind
algo.role = leduc.Roles.BigBlind
env = leduc.Game(leduc.Deck(), [player, algo])
env.deal_cards(1)
env.start()

strategy = EpsilonGreedyStrategy(eps_start,eps_end, eps_decay)
agent = Agent(strategy, device)
memory = ReplayMemory(memory_size)

policy_net = DQN()
target_net = DQN() 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_rewards = []
num_wins = 0
for episode in range(num_episodes):
    player = leduc.Player("Billy Ailish", 10)
    algo = leduc.Player("Bob", 10)
    player.role = leduc.Roles.SmallBlind
    algo.role = leduc.Roles.BigBlind
    env = leduc.Game(leduc.Deck(), [player, algo])
    env.deal_cards(1)
    env.start()

    _, state, _ = env.get_state()

    for timestep in count():
        valid_actions = env.get_actions()
        action = agent.select_action(torch.FloatTensor([state]).to(device), policy_net, valid_actions)
        env.do_action(int(action))
        reward, next_state, is_done = env.get_state()
        #reward = reward**3
        memory.push(Experience(torch.FloatTensor([state]).to(device), torch.LongTensor([action]).to(device), torch.FloatTensor([next_state]).to(device), torch.FloatTensor([reward]).to(device)))
        state = next_state
        
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions).to(device)
            next_q_values = QValues.get_next(target_net, next_states).to(device)
            target_q_values = (next_q_values * gamma) + rewards.to(device)
            
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if is_done:
            #plot?
            episode_rewards.append(reward)
            if reward>0:
                num_wins += 1
            if len(episode_rewards) % 100 == 0:
                plot(episode_rewards, 1000)
                print(sum(episode_rewards[len(episode_rewards)-1000:])/1000)
                print(f"{num_wins}%")
                num_wins = 0
            break
        
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())



#Test
num_test_episodes = 1000
wallet = num_test_episodes*10
num_test_wins = 0
win_size = np.zeros((11))
loss_size = np.zeros((10))
for episode in range(num_test_episodes):
    player = leduc.Player("Billy Ailish", 10)
    algo = leduc.Player("Bob", 10)
    player.role = leduc.Roles.SmallBlind
    algo.role = leduc.Roles.BigBlind
    env = leduc.Game(leduc.Deck(), [player, algo])
    env.deal_cards(1)
    env.start()

    for timestep in count():
        reward, state, is_done = env.get_state()
        valid_actions = env.get_actions()
        with torch.no_grad():
                out = policy_net(torch.FloatTensor([state]).to(device))
                valid_out = out[:,np.array(valid_actions)]
                idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                env.do_action(int(idx.to(device)))
            
        if is_done:
            wallet += reward
            if reward>=0:
                num_test_wins += 1
                win_size[int(reward)] += 1
            else:
                loss_size[abs(int(reward))-1] += 1
            break

print(f"Test win rate: {(num_test_wins / num_test_episodes)*100:.2F}%")
print(f"Test start wallet: {num_test_episodes*10}")
print(f"Test end wallet: {wallet}")
print(f"Won: {wallet-num_test_episodes*10}")
print(f"Win sizes: {win_size}")
print(f"Loss sizes: {loss_size}")



