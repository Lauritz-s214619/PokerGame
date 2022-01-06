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
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

class Deck:
    def __init__(self, shuffle_cards = True):
        self.cards = [Card(Suits(x), Ranks(y)) for y in range(2,15) for x in range(4)]
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
        self.is_all_in = False
    
    def show_cards(self):
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
            if (self.active_player.wallet + self.active_player.bet) > self.bet_to_call:
                actions.append(Actions.Raise)
            actions.append(Actions.Fold)
        return actions


class Game:
    def __init__(self, deck, players):
        self.deck = deck
        self.players = players
        self.active_players = players
        self.small_blind = 5
        self.big_blind = 10
        self.community_cards = []
        self.is_done = False
    
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
        #active_players = [player for player in self.players if player.in_game and not player.is_all_in]
        cycle_active_players = cycle(self.active_players)
        num_turns = 0
        while(not round.is_done):
            if len([player for player in self.active_players if not player.is_all_in]) <= 1:
                round.is_done = True
                self.is_done = True
                return

            player = next(cycle_active_players)
            if not player.in_game:
                continue
            elif player.is_all_in:
                print(f'\n{player.name} - {player.show_cards()} - Wallet: {player.wallet}')
                print(f'You\'re all in!')
                continue
            else:
                round.active_player = player
                actions = round.get_actions()
                print(f'\n{player.name} - {player.show_cards()} - Wallet: {player.wallet}')
                print(f'Your current bet: {player.bet}')
                print(f'Bet to match: {round.bet_to_call}\n')
                for action in actions:
                    print(f'{action.value}. {action.name}')
                
                action = -1
                action_completed = False
                while(action not in [action.value for action in actions] and not action_completed):
                    try:
                        action = int(input(f'Choice: '))
                    except ValueError:
                        print("Please enter an integer")

                    if action == Actions.Call.value:
                        diff = round.bet_to_call - player.bet
                        if player.wallet >= diff:
                            player.wallet -= diff
                            player.bet += diff
                        else:
                            player.bet += player.wallet
                            player.wallet = 0
                            player.is_all_in = True
                            print("All in!")
                        action_completed = True
                    elif action == Actions.Raise.value:
                        raise_completed = False
                        while(not raise_completed):
                            bet = int(input(f'Amount to bet (min {self.big_blind}): '))
                            if bet > round.bet_to_call and bet >= self.big_blind and bet <= player.wallet:
                                player.wallet -= bet
                                player.bet += bet
                                round.bet_to_call = player.bet
                                raise_completed = True
                                action_completed = True
                                if player.wallet == 0:
                                    player.is_all_in = True
                                    print("All in!")
                            else:
                                print(f'Not a valid bet. Your wallet: {player.wallet}')
                    elif action == Actions.Fold.value:
                        player.in_game = False
                        self.active_players.remove(player)
                        action_completed = True

                num_turns += 1
                #Check if round is done
                bets = [player.bet for player in self.players if player.in_game]
                if bets.count(bets[0]) == len(bets) and num_turns >= len(self.active_players):
                    round.is_done = True

        round.pot = sum(bets)
        for player in self.active_players:
            player.bet = 0
    
    def get_winner(self):
        for player in self.active_players:
            player.best_hand()

        

player1 = Player("Bob", Hand(), 100)
player2 = Player("Alice", Hand(), 100)
game1 = Game(Deck(), [player1, player2])
game1.deal_cards(2)
print(f'''
=============================
{player1.name}: {player1.show_cards()}
{player2.name}: {player2.show_cards()}
=============================
''')

print("Round 1:")
#game1.new_round()

print("\nRound 2 - Flop:")
game1.show_card(3)

#game1.new_round()

print("\nRound 3 - Turn:")
game1.show_card()

#game1.new_round()

print("\nRound 4 - River:")
game1.show_card()

#game1.new_round()

#game1.get_winner()
print("\n")

all_cards = player1.hand.cards + game1.community_cards


ranks = [card.rank.value for card in all_cards]
suits = [card.rank.value for card in all_cards]
higest_card = max(ranks)

pairs = [rank for rank in ranks if ranks.count(rank)==2]
if pairs:
    higest_pair = max(pairs)


print(pairs)
print(max([rank for rank in ranks if ranks.count(rank)>1]))
print(len([card for card in all_cards if card.suit == Suits.Spades]))
print(len([card for card in all_cards if card.suit == Suits.Hearts]))
print(len([card for card in all_cards if card.suit == Suits.Clubs]))
print(len([card for card in all_cards if card.suit == Suits.Diamonds]))
#print(card for card in all_cards if card == Card(Suits.Hearts, Ranks.Ace))
