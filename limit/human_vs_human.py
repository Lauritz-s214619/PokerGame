from limit import *
player1 = Player("Alice", 50)
player2 = Player("Bob", 50)
env = Game([player1, player2], verbose=True)
env.reset()
is_done = False
while not is_done:
    action_completed = False
    while not action_completed:
        try:
            action = int(input("Select action: "))
            action_completed = True
        except ValueError:
            print("Please enter an integer")
    env.do_action(action)
    _,_,is_done = env.get_state()
