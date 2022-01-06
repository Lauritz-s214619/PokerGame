from enum import Enum
from itertools import cycle
import random

class Suits(Enum):
    Spades   = 0
    Hearts   = 1
    Clubs    = 2
    Diamonds = 3

class Ranks(Enum):

    Two   = 2
    Three = 3
    Four  = 4
    Five  = 5
    Six   = 6
    Seven = 7
    Eight = 8
    Nine  = 9
    Ten   = 10
    Jack  = 11
    Queen = 12
    King  = 13
    Ace   = 14

class Actions(Enum):
    Check = 0
    Call  = 1
    Raise = 2
    Fold  = 3


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        return f'{self.rank.name} of {self.suit.name}'

class Deck:
    def __init__(self, shuffle_cards = True):
        self.cards = [Card(Suits(x), Ranks(y)) for y in range(1,14) for x in range(4)]
        if shuffle_cards:
            random.shuffle(self.cards)
    
    def deal_card(self):
        return self.cards.pop()


class Hand:
    def __init__(self):
        self.cards = []

class Player:
    def __init__(self, name, hand, wallet):
        self.name = name
        self.hand = hand
        self.wallet = wallet
        self.bet = 0
        self.in_game = True
    
    def get_cards(self):
        return [str(card) for card in self.hand.cards]

class Round:
    def __init__(self, players):
        self.pot = 0
        self.bet_to_call = 0
        self.active_player = players[0]
        self.is_done = False
    
    def get_actions(self):
        actions = []
        if self.active_player.bet == self.bet_to_call:
            actions.append(Actions.Check)
            actions.append(Actions.Raise)
        elif self.active_player.bet < self.bet_to_call:
            actions.append(Actions.Call)
            actions.append(Actions.Raise)
            actions.append(Actions.Fold)
        return actions


class Game:
    def __init__(self, deck, players):
        self.deck = deck
        self.players = players
        self.small_blind = 5
        self.big_blind = 10
        self.community_cards = []
        self.is_done = False
        self.num_active_players = len(players)
    
    def deal_cards(self, num = 2):
        for player in self.players:
            for i in range(num):
                player.hand.cards.append(self.deck.deal_card())            
    
    def show_card(self, num = 1):
        for i in range(num):
            card = self.deck.deal_card()
            self.community_cards.append(card)
            print(card)
    
    def new_round(self):
        round = Round(self.players)
        active_players = [player for player in self.players if player.in_game]
        cycle_active_players = cycle(active_players)
        while(not round.is_done):
            if len(self.num_active_players) == 1:
                round.is_done = True
                self.is_done = True
                break

            player = next(cycle_active_players)
            if not player.in_game:
                continue
            else:
                round.active_player = player
                actions = round.get_actions()
                print(f'\n{player.name} - {player.get_cards()} - Wallet: {player.wallet}')
                print(f'Your current bet: {player.bet}')
                print(f'Bet to match: {round.bet_to_call}\n')
                for action in actions:
                    print(f'{action.value}. {action.name}')
                
                action = -1
                while(action not in [action.value for action in actions]):
                    try:
                        action = int(input(f'Choice: '))
                    except ValueError:
                        print("Please enter an integer")

                    if action == 1:
                        if player.wallet >= round.bet_to_call:
                            diff = round.bet_to_call - player.bet
                            player.wallet -= diff
                            player.bet += diff
                    elif action == 2:
                        raise_completed = False
                        while(not raise_completed):
                            bet = int(input(f'Amount to bet (min {self.big_blind}): '))
                            if bet > round.bet_to_call and bet >= self.big_blind and bet <= player.wallet:
                                player.wallet -= bet
                                player.bet += bet
                                round.bet_to_call += bet
                                raise_completed = True
                    elif action == 3:
                        player.in_game = False
            
                #Check if round is done
                bets = [player.bet for player in self.players if player.in_game]                      
                if bets.count(bets[0]) == len(bets):
                    round.is_done = True

        round.pot = sum(bets)
        for player in active_players:
            player.bet = 0

        

player1 = Player("Bob", Hand(), 100)
player2 = Player("Alice", Hand(), 100)
game1 = Game(Deck(), [player1, player2])
game1.deal_cards(2)
print(f'''
=============================
{player1.name}: {player1.get_cards()}
{player2.name}: {player2.get_cards()}
=============================
''')

print("Round 1:")
game1.new_round()

print("\nRound 2 - Flop:")
game1.show_card(3)

game1.new_round()

print("\nRound 3 - Turn:")
game1.show_card()

game1.new_round()

print("\nRound 4 - River:")
game1.show_card()

game1.new_round()
