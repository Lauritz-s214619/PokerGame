import numpy as np
import random
from limit import *
from DQN_Agent import DQN

random_agent = Player("Rob Random", 50, show_card=False)
player = Player("Human", 50)
env = Game([random_agent, player], verbose=True)

num_games = 5
num_wins = 0
num_loss = 0
wallet = num_games*50
for _ in range(num_games):
    env.reset()
    is_done = False
    while not is_done:
        if env.active_player == random_agent:
            valid_actions = env.get_actions()
            env.do_action(random.choice([action for action in valid_actions if action != 3]))
        else:
            action_completed = False
            while not action_completed:
                try:
                    action = int(input("Select action: "))
                    action_completed = True
                except ValueError:
                    print("Please enter an integer")
            env.do_action(action)
        reward,_,is_done = env.get_state()
    wallet += reward
    if reward>0:
        num_wins += 1
    elif reward<0:
        num_loss += 1
    print(f"Reward: {reward}")

print(f"Played {num_games} games")
print(f"Won {num_wins} times")
print(f"Loss {num_loss} times")
print(f"Tie {num_games - (num_wins+num_loss)} times")
print(f"Test win rate: {(num_wins / num_games)*100:.2F}%")
print(f"Test start wallet: {num_games*50}")
print(f"Test end wallet: {wallet}")
print(f"Won: {wallet-num_games*50}")
print(f"Average reward: {(wallet-num_games*50)/num_games}")
