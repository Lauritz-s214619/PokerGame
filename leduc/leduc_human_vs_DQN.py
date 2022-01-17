import os
import torch
import numpy as np
from leduc import *
#from DQN_Agent import DQN

# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# policy_net_path = './models/policy_net.pt'

# policy_net = DQN()
# if os.path.exists(policy_net_path):
#     policy_net.load_state_dict(torch.load(policy_net_path))
#     policy_net.eval()

AIB = Player("Billy Ailishh", 10, False, False)
AI = Player("Billy Ailish", 10)
human = Player("Player", 10)
AI.role = Roles.SmallBlind
human.role = Roles.BigBlind
env = Game([AI, human], True)


num_episodes = 10
wallet_human = num_episodes*10
num_wins = 0
for episode in range(num_episodes):
    env.reset()

    is_done = False
    while not is_done:
        valid_actions = env.get_actions()
        reward, state, is_done = env.get_state()
        print(state)

        if is_done:
            if human in env.winners:
                wallet_human += reward
            break

        if env.active_player == AIB:
            with torch.no_grad():
                    out = policy_net(torch.FloatTensor([state]).to(device))
                    valid_out = out[:,np.array(valid_actions)]
                    idx = (out==max(valid_out[0])).nonzero(as_tuple=True)[1]
                    action = int(idx.to(device))
                    env.do_action(action)
                    #print(f"AI action: {Actions(action).name}")
        else:
            action_completed = False
            while not action_completed:
                try:
                    action = int(input("Select action: "))
                    action_completed = True
                except ValueError:
                    print("Please enter an integer")
            env.do_action(action)

print(f"Start wallet: {num_episodes*10}")
print(f"End wallet: {wallet_human}")
print(f"Won: {wallet_human-num_episodes*10}")