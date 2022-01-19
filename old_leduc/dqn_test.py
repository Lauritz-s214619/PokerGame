import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import leduc

#%% Parameters
learning_rate = 0.0001
buffer_size = 1000
num_games = 10000
num_games_stop_explore = 9900
epsilon = 1.0
epsilon_step = 2/num_games
epsilon_min = 0.05
batch_size = 64
gamma = 0.99

#%% Environment
#env = gym.make('LunarLander-v2')
player = leduc.Player("Billy Ailish", 10)
algo = leduc.Player("Bob", 10)
player.role = leduc.Roles.SmallBlind
algo.role = leduc.Roles.BigBlind
env = leduc.Game(leduc.Deck(), [player, algo])
env.deal_cards(1)
env.start()
actions = np.arange(4)

#%% Neural network
q_net = torch.nn.Sequential(
    torch.nn.Linear(16, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 4))
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()

#%% Buffers
obs_buffer = np.zeros((buffer_size, 16))
obs_next_buffer = np.zeros((buffer_size, 16))
action_buffer = np.zeros(buffer_size)
reward_buffer = np.zeros(buffer_size)
done_buffer = np.zeros(buffer_size)

#%% Training loop
step_count = 0
scores = []
for i in range(num_games):
    print(f'Playing game {i}')
    # Reset env
    player = leduc.Player("Billy Ailish", 10)
    algo = leduc.Player("Bob", 10)
    player.role = leduc.Roles.SmallBlind
    algo.role = leduc.Roles.BigBlind
    env = leduc.Game(leduc.Deck(), [player, algo])
    env.deal_cards(1)
    env.start()
    score = 0
    done = False
    _, state, _ = env.get_state()
    
    # Update epsilon
    epsilon = np.maximum(epsilon-epsilon_step, epsilon_min)
    # No exploration at the end
    if i > num_games_stop_explore:
        epsilon = 0.
    
    # Game step loop
    while not done:
        
        # Choose action
        valid_actions = env.get_actions()
        if np.random.rand() < epsilon:
            action = np.random.choice(valid_actions)
        else:
            out = q_net(torch.tensor(state).float()).detach().numpy()
            action = max([(el,i) for i,el in enumerate(out) if i in valid_actions])[1]
        
        # Step environment
        env.do_action(action)
        reward, next_state, done = env.get_state()
        score += reward
        
        # Update buffers
        obs_index = step_count % buffer_size
        obs_buffer[obs_index] = state
        obs_next_buffer[obs_index] = next_state
        action_buffer[obs_index] = action
        reward_buffer[obs_index] = reward
        done_buffer[obs_index] = done
        
        # Update observation
        observation = next_state
        
        # Learn using minibatch
        if step_count > buffer_size:
            # Choose minibatch
            batch_idx = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
            obs_batch = torch.tensor(obs_buffer[batch_idx]).float()
            obs_next_batch = torch.tensor(obs_next_buffer[batch_idx]).float()
            action_batch = action_buffer[batch_idx]
            reward_batch = torch.tensor(reward_buffer[batch_idx]).float()
            done_batch = torch.tensor(done_buffer[batch_idx]).float()
            
            # Compute loss
            q_val = q_net(obs_batch)[np.arange(batch_size), action_batch]
            target = reward_batch + gamma*torch.max(q_net(obs_next_batch), dim=1).values
            target[done_batch==1] = 0
            target.detach() 
            l = loss(q_val, target)
            
            # Step the optimizer
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # Update step count
        step_count += 1
        
    # Store game scores
    scores.append(score)
    
    # Print scores
    print(f'Score = {score}')

    # Plot scores
    if (i+1) % 100 == 0:
        print("here")
        plt.plot(scores, '.')
        plt.title(f'Steps = {step_count}, Eps = {epsilon:.2}')
        plt.pause(0.001)
        
    

























