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


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
policy_net_path = './models/policy_net.pt'

policy_net = DQN()
if os.path.exists(policy_net_path):
    policy_net.load_state_dict(torch.load(policy_net_path))
    policy_net.eval()

player = Player("Billy Ailish", 50)
random_agent = Player("Bob", 50, random=True)
env = Game([player, random_agent], verbose=False)

#Test

num_test_episodes = 10000
wallet = num_test_episodes*50
num_test_wins = 0
num_test_loss = 0
num_actions = [0]*4
rewards = []
for episode in range(num_test_episodes):
    env.reset()

    for timestep in count():
        reward, state, is_done = env.get_state()
        valid_actions = env.get_actions()
        if env.active_player == random_agent:
            # if 2 in valid_actions:
            #     env.do_action(2)
            # else:
            #     env.do_action(1)
            env.do_action(random.choice([action for action in valid_actions if action != 3]))
            reward, next_state, is_done = env.get_state()
        else:
            with torch.no_grad():
                    out = policy_net(torch.FloatTensor([state]).to(device))
                    valid_out = out[:,np.array(valid_actions)]
                    idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                    action = int(idx.to(device))
                    env.do_action(action)
                    num_actions[action] += 1
            
        if is_done:
            wallet += reward
            rewards.append(reward)
            if reward>0:
                num_test_wins += 1
            elif reward<0:
                num_test_loss += 1
            break
    if episode % 1000 == 0:
        print(f'episode: {episode} win/loss diff: {num_test_wins-num_test_loss}')

print(f"Played {num_test_episodes} games")

print(f"Won {num_test_wins} times")
print(f"Loss {num_test_loss} times")
print(f"Tie {num_test_episodes - (num_test_wins+num_test_loss)} times")
print(f"Test win rate: {(num_test_wins / num_test_episodes)*100:.2F}%")
print(f"Test start wallet: {num_test_episodes*50}")
print(f"Test end wallet: {wallet}")
print(f"Won: {wallet-num_test_episodes*50}")
print(f"Average reward: {(wallet-num_test_episodes*50)/num_test_episodes}")


print(f"Std: {np.std(rewards)}")

print(f"Number of checks: {num_actions[0]}")
print(f"Number of calls: {num_actions[1]}")
print(f"Number of raises: {num_actions[2]}")
print(f"Number of folds: {num_actions[3]}")



