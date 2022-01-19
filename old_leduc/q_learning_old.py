import numpy as np
from leduc_old import *

player = Player("Billy Ailish", 10)
algo = Player("Bob", 10)
player.role = Roles.SmallBlind
algo.role = Roles.BigBlind
env = Game(Deck(), [player, algo])
env.deal_cards(1)
env.start()

action_space_size = 4
state_space_size = 6*5*4

q_table = np.zeros((6,6,4,4))
print(q_table.size)

num_episodes = 100000


learning_rate = 0.2
discount_rate = 0.99


exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


rewards_all_episodes = []
num_wins = 0

for episode in range(num_episodes):
    player = Player("Billy Ailish", 10)
    algo = Player("Bob", 10)
    player.role = Roles.SmallBlind
    algo.role = Roles.BigBlind
    env = Game(Deck(), [player, algo])
    env.deal_cards(1)
    env.start()
    _, state, _ = env.get_state()
    
    done = False
    rewards_current_episode = 0

    while not done:


        exploration_rate_threshold = random.uniform(0,1)
        valid_actions = np.array(env.get_actions())
        if exploration_rate_threshold > exploration_rate:
            actions = q_table[state[0]-1,state[1]-1,state[2]-1,np.array([valid_actions])-1][0]
            action = valid_actions[actions==max(actions)][0]
        else:
            action = random.choice(valid_actions)

        env.do_action(action)
        reward, new_state, done = env.get_state()
        if reward>0:
            num_wins += 1
        q_table[state[0]-1, state[1]-1, state[2]-1, action-1] = q_table[state[0]-1, state[1]-1, state[2]-1, action-1] * (1-learning_rate) + learning_rate*(reward + discount_rate*np.max(q_table[new_state[0]-1,new_state[1]-1,new_state[2]-1, :]))

        state = new_state
        rewards_current_episode += reward

        if done:
            break
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)


rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("Avereage rewards per thousand episodes:")
for r in rewards_per_thousand_episodes:
    print(f'{count} : {str(sum(r/1000))}')
    count += 1000

# print(f"Win rate: {num_wins/num_episodes*100:.2F}%")

# print("Final Q-table:")
#print(q_table)


#Testing the q-table
num_test_episodes = 1000
num_test_wins = 0
wallet = 10*num_test_episodes
for episode in range(num_test_episodes):
    player = Player("Billy Ailish", 10)
    algo = Player("Bob", 10)
    player.role = Roles.SmallBlind
    algo.role = Roles.BigBlind
    env = Game(Deck(), [player, algo])
    env.deal_cards(1)
    env.start()
    
    done = False
    rewards_current_episode = 0

    while not done:
        reward, state, done = env.get_state()
        wallet += reward

        valid_actions = np.array(env.get_actions())

        actions = q_table[state[0]-1,state[1]-1,state[2]-1,np.array([valid_actions])-1][0]
        action = valid_actions[actions==max(actions)][0]

        env.do_action(action)
        # if 3 in valid_actions:
        #     env.do_action(3)
        # elif 2 in valid_actions:
        #     env.do_action(2)
        # else:
        #     env.do_action(1)

        if reward>0:
            num_test_wins += 1

        if done:
            break

print(f"Test win rate: {num_test_wins/num_test_episodes*100:.2F}%")
print(f"Won: {wallet-10*num_test_episodes}")