import numpy as np
from leduc import *
from leduc import Actions

def get_rank(card):
    if card in [0,3]:
        return 0
    elif card in [1,4]:
        return 1
    else:
        return 2

def check_is_done(env):
    _, _, is_done = env.get_state()
    return is_done

player = Player("Human", 10)
algo = Player("Algo", 10, True)
player.role = Roles.SmallBlind
algo.role = Roles.BigBlind
env = Game([player, algo], verbose=False)

num_test_episodes = 1000000
wallet = num_test_episodes*10
num_test_wins = 0
for _ in range(num_test_episodes):
    env.reset()

    reward, state, is_done = env.get_state()
    my_rank = get_rank(np.argmax(state[:6]))
    last_action = None
    if max(state[12:])>0:
        last_action = np.argmax(state[12:])
    
    if last_action == Actions.Raise.value:
        algo_rank = 2
        #env.do_action(Actions.Call.value)
        env.do_action(Actions.Fold.value)
        wallet+=reward
        continue

    else:
        env.do_action(Actions.Raise.value)

        reward, state, is_done = env.get_state()
        last_action = np.argmax(state[12:])
    
        if last_action == Actions.Fold.value:
            wallet+=reward
            #print(f"WALLET {reward}")
            if not is_done:
                print(is_done)
            continue
        elif last_action == Actions.Call.value:
            algo_rank = 1
        elif last_action == Actions.Raise.value:
            algo_rank = 2
            env.do_action(Actions.Call.value)
    
    _, state, is_done = env.get_state()
    last_action = np.argmax(state[12:])
    community_rank = get_rank(np.argmax(state[6:12]))

    if my_rank == community_rank or (algo_rank != community_rank and my_rank>=algo_rank):
        is_done = False
        while not is_done:
            valid_actions = env.get_actions()
            if Actions.Raise.value in valid_actions:
                env.do_action(Actions.Raise.value)
            else:
                env.do_action(Actions.Call.value)
            _, _, is_done = env.get_state()
    else:
        is_done = False
        while not is_done:
            valid_actions = env.get_actions()
            if Actions.Fold.value in valid_actions:
                env.do_action(Actions.Fold.value)
            else:
                env.do_action(Actions.Check.value)
            _, _, is_done = env.get_state()
    
    reward, state, is_done = env.get_state()
    if not is_done:
        print(is_done)
    wallet+=reward
    #print(f"WALLET {reward}")



print(f"Test win rate: {(num_test_wins / num_test_episodes)*100:.2F}%")
print(f"Test start wallet: {num_test_episodes*10}")
print(f"Test end wallet: {wallet}")
print(f"Won: {wallet-num_test_episodes*10}")
print(f"Average reward: {(wallet-num_test_episodes*10)/num_test_episodes}")

