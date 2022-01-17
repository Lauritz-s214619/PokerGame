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

AI = Player("Billy Ailish", 50)
player = Player("Human", 50)
env = Game([AI, player], verbose=True)

num_games = 10
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
        _,_,is_done = env.get_state()
