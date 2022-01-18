import os
import torch
import numpy as np
from limit import *
from DQN_Agent import DQN

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
policy_net_path = './models/policy_net.pt'

policy_net = DQN()
if os.path.exists(policy_net_path):
    policy_net.load_state_dict(torch.load(policy_net_path))
    policy_net.eval()

AI = Player("Billy Ailish", 50, show_card=False)
player = Player("Human", 50)
env = Game([AI, player], verbose=True)

num_games = 5
num_wins = 0
num_loss = 0
wallet = num_games*50
for _ in range(num_games):
    env.reset()
    is_done = False
    while not is_done:
        if env.active_player == AI:
            _, state, _ = env.get_state()
            valid_actions = env.get_actions()
            with torch.no_grad():
                out = policy_net(torch.FloatTensor([state]).to(device))
                valid_out = out[:,np.array(valid_actions)]
                idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                env.do_action(int(idx.to(device)))
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
