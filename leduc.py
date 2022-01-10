from enum import Enum, unique
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

class PokerHand(Enum):
    HighestCard = 0
    Pair  = 1
    TwoPair = 2
    ThreeOfAKind  = 3
    Straight  = 4
    Flush  = 5
    FullHouse  = 6
    FourOfAKind  = 7
    StraightFlush  = 8
    RoyalFlush  = 9

class Roles(Enum):
    Player = 0
    Button = 1
    SmallBlind = 2
    BigBlind = 3

class Card:
    def __init__(self, suit, rank, id):
        self.suit = suit
        self.rank = rank
        self.id = id
    
    def __str__(self):
        return f'{self.rank.name} of {self.suit.name}'
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

class Deck:
    def __init__(self, shuffle_cards = True):
        self.cards = [Card(Suits(x), Ranks(y), y-10+x*3) for y in range(11,14) for x in range(2)]
        if shuffle_cards:
            random.shuffle(self.cards)
    
    def deal_card(self):
        return self.cards.pop()

class Player:
    def __init__(self, name, wallet):
        self.name = name
        self.card = None
        self.wallet = wallet
        self.bet = 0
        self.in_game = True
        self.is_all_in = False
        self.role = Roles.Player
    
    def show_cards(self):
        return [str(card) for card in self.hand.cards]

class Round:
    def __init__(self, bet_to_call = 0):
        self.pot = 0
        self.bet_to_call = bet_to_call
        self.active_player = None
        self.is_done = False
        self.num_actions = 0
        self.last_action = None
    
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
        self.small_blind = 1
        self.big_blind = 1
        self.community_card = 0
        self.is_done = False
        self.is_pre_flop = True
        self.pot = 0
        self.winners = []
        self.num_rounds = 0

        for player in self.players:
            if player.role == Roles.SmallBlind:
                if player.wallet >= self.small_blind:
                    player.wallet -= self.small_blind
                    player.bet = self.small_blind
                else:
                    print("Error, wallet size is to samll for small blind")
            elif player.role == Roles.BigBlind:
                if player.wallet >= self.big_blind:
                    player.wallet -= self.big_blind
                    player.bet = self.big_blind
                else:
                    print("Error, wallet size is to samll for big blind")
    
    def deal_cards(self, num = 1):
        for player in self.players:
            for _ in range(num):
                player.card = self.deck.deal_card()
    
    def show_card(self, num = 1):
        for _ in range(num):
            card = self.deck.deal_card()
            self.community_card = card
            print(card)
    
    def new_round(self, bet_to_call = 0):
        self.num_rounds += 1
        self.round = Round(bet_to_call)
    
    def start(self):
        self.new_round(1)
        if random.randint(0, 1):
            self.round.active_player = self.players[1]
            if self.players[1].card.rank.value < Ranks.King.value:
                self.do_action(Actions.Check.value)
            else:
                self.do_action(Actions.Raise.value)
        else:
            self.round.active_player = self.players[0]
    
    def do_action(self, action):
        valid_actions = self.round.get_actions()
        if action in valid_actions:

            if action == Actions.Check.value:
                self.last_action = Actions.Check.value

            elif action == Actions.Call.value:
                self.round.active_player.wallet -= 1
                self.round.active_player.bet += 1
                self.round.pot += 1
                self.last_action = Actions.Call.value

            elif action == Actions.Raise.value:
                self.round.bet_to_call += 1
                bet = self.round.bet_to_call - self.round.active_player.bet
                self.round.active_player.wallet -= bet
                self.round.active_player.bet += bet
                self.round.pot += bet
                self.last_action = Actions.Raise.value

            elif action == Actions.Fold.value:
                self.is_done = True
                self.last_action = Actions.Fold.value
                self.winners = [player for player in self.players if player != self.round.active_player]
                self.get_winner()
        else:
            print("Error, invalid action!")
        
        self.round.num_actions += 1
        if self.round.num_actions >= 2:
            round_done = True
            for player in self.players:
                if player.bet != self.round.bet_to_call:
                    round_done = False
                    break
            if round_done:
                self.pot += self.round.pot
                for player in self.players:
                    player.bet = 0
                
                if self.num_rounds == 2:
                    self.is_done = True
                    self.get_winner()
                else:
                    self.show_card()
                    self.new_round()

        self.round.active_player = [player for player in self.players if player != self.round.active_player][0]

        if self.round.active_player == self.players[1]:
            self.do_algo_action()
    
    def do_algo_action(self):
        if self.num_rounds == 1:
            if self.players[1].card.rank == Ranks.Jack:
                if self.round.last_action == Actions.Raise:
                    self.do_action(Actions.Fold)
                else:
                    self.do_action(Actions.Check)

            elif self.players[1].card.rank == Ranks.Queen:
                if self.round.last_action == Actions.Raise:
                    self.do_action(Actions.Call)
                else:
                    self.do_action(Actions.Check)
            
            else:
                self.do_action(Actions.Raise)
        else:
            if self.players[1].card.rank == self.community_card.rank:
                self.do_action(Actions.Raise)
            else:
                if self.players[1].card.rank == Ranks.Jack:
                    if self.round.last_action == Actions.Raise:
                        self.do_action(Actions.Fold)
                    else:
                        self.do_action(Actions.Check)

                elif self.players[1].card.rank == Ranks.Queen:
                    if self.round.last_action == Actions.Raise:
                        self.do_action(Actions.Fold)
                    else:
                        self.do_action(Actions.Check)

                else:
                    if self.round.last_action == Actions.Raise:
                        self.do_action(Actions.Call)
                    elif self.round.last_action == Actions.Check:
                        self.do_action(Actions.Raise)
                    else:
                        self.do_action(Actions.Check)
    
    def get_winner(self):
        if not self.winners:
            if self.players[0].card.rank.value > self.players[1].card.rank.value:
                self.winners = self.players[0]

            elif self.players[0].card.rank.value < self.players[1].card.rank.value:
                self.winners = self.players[1]

            else:
                self.winners = self.players
            
            for player in self.players:
                if player.card == self.community_card:
                    self.winners = [player]
                    break
                
        if len(self.winners) > 1:
            print("It's a tie!")
            self.winners[0].wallet += self.pot/2
            self.winners[1].wallet += self.pot/2
        else:
            self.winners[0].wallet += self.pot
            print(f'{self.winner.name} has won!')
    
    def get_state(self):
        reward = 0
        if self.is_done:
            reward = self.players[0].wallet - 10
        return reward, [self.players[0].card.id, self.community_card, self.round.last_action], self.is_done


        





player1 = Player("Bob", 10)
player2 = Player("Alice", 10) #Algoritme
player1.Role = Roles.SmallBlind
player2.Role = Roles.BigBlind
game = Game(Deck(), [player1, player2])
game.deal_cards(1)
game.start()
game.get_state()
game.do_action()