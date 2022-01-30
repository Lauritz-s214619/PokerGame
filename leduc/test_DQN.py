import math
import random
from sched import scheduler
from unittest.mock import seal
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
from leduc import *
from leduc_DQN_Agent import *


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
policy_net_path = './models/policy_net.pt'

policy_net = DQN()
if os.path.exists(policy_net_path):
    policy_net.load_state_dict(torch.load(policy_net_path))
    policy_net.eval()

player = Player("Billy Ailish", 50)
random_agent = Player("Bob", 50, is_algo=True)
env = Game([player, random_agent], verbose=False)

#Test

num_episodes = 10000
wallet = num_episodes*10
num_wins = 0
num_loss = 0
num_actions = [0]*4
num_jack_actions = [0]*4
num_queen_actions = [0]*4
num_king_actions = [0]*4
num_pairs_actions = [0]*4
num_pairs_when_fold = 0
num_ranks_when_fold = [0]*3
num_pairs = 0
rewards = []
for episode in range(num_episodes):
    env.reset()

    for timestep in count():
        reward, state, is_done = env.get_state()
        valid_actions = env.get_actions()
        with torch.no_grad():
                out = policy_net(torch.FloatTensor([state]).to(device))
                valid_out = out[:,np.array(valid_actions)]
                idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                action = int(idx.to(device))
                env.do_action(action)
                num_actions[action] += 1

                if env.community_card:
                    if player.card.rank == env.community_card.rank:
                        num_pairs_actions[action] += 1
                    else:
                        if player.card.rank == Ranks.Jack:
                            num_jack_actions[action] += 1
                        elif player.card.rank == Ranks.Queen:
                            num_queen_actions[action] += 1
                        elif player.card.rank == Ranks.King:
                            num_king_actions[action] += 1
                    
                    
                    if action == 3 and player.card.rank == env.community_card.rank:
                        num_pairs_when_fold += 1
                else:
                    if player.card.rank == Ranks.Jack:
                        num_jack_actions[action] += 1
                    elif player.card.rank == Ranks.Queen:
                        num_queen_actions[action] += 1
                    elif player.card.rank == Ranks.King:
                        num_king_actions[action] += 1
                
                if action == 3:
                    num_ranks_when_fold[player.card.rank.value-11] += 1
            
        if is_done:
            if env.community_card:
                if player.card.rank == env.community_card.rank:
                    num_pairs += 1
            wallet += reward
            rewards.append(reward)
            if reward>0:
                num_wins += 1
            elif reward<0:
                num_loss += 1
            else:
                num_wins+=0.5
            break
    if episode % 10000 == 0:
        print(episode)

print(f"Played {num_episodes} games")

print(f"Won {num_wins} times")
print(f"Loss {num_loss} times")
print(f"Tie {num_episodes - (num_wins+num_loss)} times")
print(f"Test win rate: {(num_wins / num_episodes)*100:.2F}%")
print(f"Test start wallet: {num_episodes*10}")
print(f"Test end wallet: {wallet}")
print(f"Won: {wallet-num_episodes*10}")
print(f"Average reward: {(wallet-num_episodes*10)/num_episodes}")


print(f"Std: {np.std(rewards)}")
print(f"Var: {np.var(rewards)}")

print(f"Number of checks: {num_actions[0]}")
print(f"Number of calls: {num_actions[1]}")
print(f"Number of raises: {num_actions[2]}")
print(f"Number of folds: {num_actions[3]}")
print(num_pairs)
print(num_pairs_when_fold)
print(num_ranks_when_fold)
print("Here: ")
print(num_jack_actions)
print(num_queen_actions)
print(num_king_actions)
print(num_pairs_actions)

print([num/sum(num_jack_actions)*100 for num in num_jack_actions])
print([num/sum(num_queen_actions)*100 for num in num_queen_actions])
print([num/sum(num_king_actions)*100 for num in num_king_actions])
print([num/sum(num_pairs_actions)*100 for num in num_pairs_actions])



