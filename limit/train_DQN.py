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
import os
from limit import *
from DQN_Agent import *


policy_net_path = './models/policy_net.pt'
target_net_path = './models/target_net.pt'

## Plot function
def plot(values, period):
    #plt.figure(2)
    plt.clf()
    plt.title(f'Training, Eps: {agent.eps}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-3,3)
    #plt.plot(values)
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
    t5 = [val.cpu().numpy()[0].astype(int) for val in batch.valid_actions]
    
    return (t1,t2,t3,t4,t5)


## Parametre

batch_size = 32
gamma = 0.99    
eps_start = 1  
eps_end = 0.1      
eps_decay = 0.0002
eps_steps = 10000
target_update = 1000
memory_size = 10000
lr = 0.00005
num_episodes = 20000

##Main Program

player = Player("Billy Ailish", 50)
random_agent = Player("Bob", 50, random=True)
env = Game([player, random_agent], verbose=False)

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay, eps_steps)
agent = Agent(strategy, device)
memory = ReplayMemory(memory_size)

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.99)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0, verbose=True)
#scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer)
episode_rewards = []
num_wins = 0
for episode in range(num_episodes):
    env.reset()

    _, state, _ = env.get_state()

    for timestep in count():
        valid_actions = env.get_actions()
        if env.active_player == random_agent:
            env.do_action(random.choice([action for action in valid_actions if action != 3]))
            reward, next_state, is_done = env.get_state()
        else:
            action = agent.select_action(torch.FloatTensor([state]).to(device), policy_net, valid_actions)
            env.do_action(int(action))
            reward, next_state, is_done = env.get_state()
            new_valid_actions = env.get_actions()
            #reward = -reward
            #reward = reward**3
            memory.push(Experience(torch.FloatTensor([state]).to(device), torch.LongTensor([action]).to(device), torch.FloatTensor([next_state]).to(device), torch.FloatTensor([reward]).to(device), torch.FloatTensor([new_valid_actions]).to(device)))
            state = next_state
            
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states, valid_actions = extract_tensors(experiences)
                current_q_values = QValues.get_current(policy_net, states, actions).to(device)
                next_q_values = QValues.get_next(target_net, next_states, valid_actions).to(device)
                target_q_values = (next_q_values * gamma) + rewards.to(device)
                
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()
                #scheduler.step(loss)
            
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

#Save models
torch.save(policy_net.state_dict(), './models/policy_net.pt')
torch.save(target_net.state_dict(), './models/target_net.pt')



#Test
num_test_episodes = 10000
wallet = num_test_episodes*50
num_test_wins = 0
for episode in range(num_test_episodes):
    env.reset()

    for timestep in count():
        reward, state, is_done = env.get_state()
        valid_actions = env.get_actions()
        if env.active_player == random_agent:
            env.do_action(random.choice([action for action in valid_actions if action != 3]))
            reward, next_state, is_done = env.get_state()
        else:
            with torch.no_grad():
                    out = policy_net(torch.FloatTensor([state]).to(device))
                    valid_out = out[:,np.array(valid_actions)]
                    idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                    env.do_action(int(idx.to(device)))
            
        if is_done:
            wallet += reward
            if reward>=0:
                num_test_wins += 1
            break

print(f"Test win rate: {(num_test_wins / num_test_episodes)*100:.2F}%")
print(f"Test start wallet: {num_test_episodes*50}")
print(f"Test end wallet: {wallet}")
print(f"Won: {wallet-num_test_episodes*50}")
print(f"Average reward: {(wallet-num_test_episodes*50)/num_test_episodes}")
input("Press any button to close the plot...")



