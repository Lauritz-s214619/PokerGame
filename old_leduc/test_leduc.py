from leduc import *

p1 = Player("Player 1", 10)
p2 = Player("Player 2", 10)
p1.role = Roles.SmallBlind
p2.role = Roles.BigBlind
env = Game([p1, p2])
env.reset()

while True:
    env.reset()
    done = False
    print("New round!")
    while not done:
        print(f"Round number: {env.num_rounds}")
        print(f'{env.active_player.name} - {env.active_player.card} - Wallet: {env.active_player.wallet}')
        print(f'Pot: {env.pot}')
        if env.community_card:
            print(f"Community card: {env.community_card.suit.name} {env.community_card.rank.name}")
        
        print(env.get_actions())
        action = int(input("Select action: "))
        env.do_action(action)
        _, _, done = env.get_state()

