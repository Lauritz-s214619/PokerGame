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
        self.cards = [Card(Suits(x), Ranks(y), y-11+x*3) for y in range(11,14) for x in range(2)]
        if shuffle_cards:
            random.shuffle(self.cards)
    
    def deal_card(self):
        return self.cards.pop()

class Player:
    def __init__(self, name, wallet, is_algo = False, show_card = True):
        self.name = name
        self.card = None
        self.wallet = wallet
        self.bet = 0
        self.in_game = True
        self.is_all_in = False
        self.role = Roles.Player
        self.is_algo = is_algo
        self.show_card = show_card
    
    def show_cards(self):
        return [str(card) for card in self.hand.cards]
    
    def __eq__(self, other):
        if not isinstance(other, Player):
            return NotImplemented
        return self.name == other.name

class Round:
    def __init__(self, bet_to_call = 0):
        self.pot = 0
        self.bet_to_call = bet_to_call
        self.is_done = False
        self.num_actions = 0
        self.last_action = None
        self.num_raises = 0

class Game:
    def __init__(self, players, verbose = False):
        self.deck = Deck()
        self.players = players
        self.active_player = None
        self.small_blind = 1
        self.big_blind = 1
        self.community_card = 0
        self.is_done = False
        self.is_pre_flop = True
        self.pot = self.small_blind + self.big_blind
        self.winners = []
        self.num_rounds = 0
        self.verbose = verbose
        self.raise_history = [1,0,0,1,0,0]
        self.last_action = None

    def get_actions(self):
        actions = []
        if self.active_player.bet == self.round.bet_to_call:
            actions.append(Actions.Check)
            actions.append(Actions.Raise)
        elif self.active_player.bet < self.round.bet_to_call:
            actions.append(Actions.Call)
            if (self.active_player.wallet + self.active_player.bet) > self.round.bet_to_call and self.round.num_raises < 2:
                actions.append(Actions.Raise)
            actions.append(Actions.Fold)
        return [action.value for action in actions]

    
    def deal_cards(self, num = 1):
        for player in self.players:
            for _ in range(num):
                player.card = self.deck.deal_card()
    
    def show_card(self, num = 1):
        for _ in range(num):
            card = self.deck.deal_card()
            self.community_card = card
            #print(card)
    
    def new_round(self, bet_to_call = 0):
        self.num_rounds += 1
        self.round = Round(bet_to_call)
        if self.verbose:
            print(f"\nRound {self.num_rounds}")
    
    def reset(self):
        if self.verbose:
            print("\nNew game!")
        self.deck = Deck()
        self.active_player = None
        self.community_card = 0
        self.is_done = False
        self.is_pre_flop = True
        self.pot = self.small_blind + self.big_blind
        self.winners = []
        self.num_rounds = 0
        self.raise_history = [1,0,0,1,0,0]

        for player in self.players:
            player.wallet = 10
        
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

        self.deal_cards(1)

        self.new_round(1)
        if random.randint(0, 1):
            self.active_player = self.players[1]
        else:
            self.active_player = self.players[0]

        if self.verbose:
            self.print_info()

        if self.active_player.is_algo:
                self.do_algo_action()
    
    def do_action(self, action):
        if self.verbose:
            print(f"{Actions(action).name}!")
        valid_actions = self.get_actions()
        if action in valid_actions:

            if action == Actions.Check.value:
                self.last_action = Actions.Check.value

            elif action == Actions.Call.value:
                self.active_player.wallet -= 1
                self.active_player.bet += 1
                self.pot += 1
                self.last_action = Actions.Call.value

            elif action == Actions.Raise.value:
                self.round.bet_to_call += 1
                self.round.num_raises += 1
                bet = self.round.bet_to_call - self.active_player.bet
                self.active_player.wallet -= bet
                self.active_player.bet += bet
                self.pot += bet
                self.last_action = Actions.Raise.value
                self.raise_history[3*(self.num_rounds-1):3*self.num_rounds] = [0]*3
                self.raise_history[self.round.num_raises+3*(self.num_rounds-1)] = 1

            elif action == Actions.Fold.value:
                self.is_done = True
                self.last_action = Actions.Fold.value
                self.winners = [player for player in self.players if player != self.active_player]
                self.get_winner()
                return
        else:
            print(f"Error, invalid action! {valid_actions} {action} {self.active_player.name}")
        
        self.round.num_actions += 1
        if self.round.num_actions >= 2:
            round_done = True
            for player in self.players:
                if player.bet != self.round.bet_to_call:
                    round_done = False
                    break
            if round_done:
                for player in self.players:
                    player.bet = 0
                
                if self.num_rounds == 2:
                    self.is_done = True
                    self.get_winner()
                else:
                    self.show_card()
                    self.new_round()

        self.active_player = [player for player in self.players if player != self.active_player][0]
        
        if self.verbose and not self.is_done:
            self.print_info()

        if self.active_player.is_algo and not self.is_done:
            self.do_algo_action()
    
    def do_algo_action(self):
        if self.num_rounds == 1:
            if self.players[1].card.rank == Ranks.Jack:
                if self.last_action == Actions.Raise.value:
                    self.do_action(Actions.Fold.value)
                else:
                    self.do_action(Actions.Check.value)

            elif self.players[1].card.rank == Ranks.Queen:
                if self.last_action == Actions.Raise.value:
                    self.do_action(Actions.Call.value)
                else:
                    self.do_action(Actions.Check.value)
            
            else:
                if self.round.num_raises < 2:
                    self.do_action(Actions.Raise.value)
                else:
                    self.do_action(Actions.Call.value)
        else:
            if self.players[1].card.rank == self.community_card.rank:
                if self.round.num_raises < 2:
                    self.do_action(Actions.Raise.value)
                else:
                    self.do_action(Actions.Call.value)
            else:
                if self.players[1].card.rank == Ranks.Jack:
                    if self.last_action == Actions.Raise.value:
                        self.do_action(Actions.Fold.value)
                    else:
                        self.do_action(Actions.Check.value)

                elif self.players[1].card.rank == Ranks.Queen:
                    if self.last_action == Actions.Raise.value:
                        self.do_action(Actions.Fold.value)
                    else:
                        self.do_action(Actions.Check.value)

                else:
                    if self.last_action == Actions.Raise.value:
                        self.do_action(Actions.Call.value)
                    elif self.last_action == Actions.Check.value:
                        if self.round.num_raises < 2:
                            self.do_action(Actions.Raise.value)
                        else:
                            self.do_action(Actions.Call.value)
                    else:
                        self.do_action(Actions.Check.value)
    
    def get_winner(self):
        if not self.winners:
            if self.players[0].card.rank.value > self.players[1].card.rank.value:
                self.winners = [self.players[0]]

            elif self.players[0].card.rank.value < self.players[1].card.rank.value:
                self.winners = [self.players[1]]

            else:
                self.winners = self.players
            
            for player in self.players:
                if player.card.rank == self.community_card.rank:
                    self.winners = [player]
                    break
                
        if len(self.winners) > 1:
            if self.verbose:
                print("\nIt's a tie!")
            self.winners[0].wallet += self.pot/2
            self.winners[1].wallet += self.pot/2
        else:
            self.winners[0].wallet += self.pot
            if self.verbose:
                print(f'\n{self.winners[0].name} has won!')
        if self.verbose:
            print(f'{self.players[0].name} card: {self.players[0].card}')
            print(f'{self.players[1].name} card: {self.players[1].card}')
    
    def get_state(self):
        reward = 0
        if self.is_done:
            reward = self.players[0].wallet - 10
        
        state = [0]*16 #[0]*18 # 0-5, player card, 6-11 community card, 12-14 num raises in round 1, 15-17 num raises in round 2
        state[self.players[0].card.id] = 1
        if self.community_card:
            state[self.community_card.id+6] = 1
        if self.last_action is not None:
            #print("HERE")
            #print(self.last_action)
            state[self.last_action+12] = 1
        #state[12:] = self.raise_history


        return reward, state, self.is_done
    
    def print_info(self):
        valid_actions = self.get_actions()
        if self.active_player.show_card:
            print(f'\n{self.active_player.name} - {self.active_player.card} - Wallet: {self.active_player.wallet} - Pot: {self.pot}')
            print(f'Bet to call: {self.round.bet_to_call}')
            if self.community_card:
                print(f"Community card: {self.community_card}")
            print("\n".join(map(lambda x: f"{x}: {Actions(x).name}", valid_actions)))
        else:
            print(f'\n{self.active_player.name} - Wallet: {self.active_player.wallet}')