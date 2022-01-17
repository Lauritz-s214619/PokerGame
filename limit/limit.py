from enum import Enum, unique
from itertools import cycle
import random

class Suits(Enum):
    Spades   = 0
    Hearts   = 1
    Clubs    = 2
    Diamonds = 3

class Ranks(Enum):
    Ace   = 1
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
        return pretty_card(self) #f'{self.rank.name} of {self.suit.name}'
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

class Deck:
    def __init__(self, shuffle_cards = True):
        self.cards = [Card(Suits(x), Ranks(y), y-1+x*13) for y in range(1,14) for x in range(4)]
        #print([card.id for card in self.cards])
        if shuffle_cards:
            random.shuffle(self.cards)
    
    def deal_card(self):
        return self.cards.pop()

class Player:
    def __init__(self, name, wallet, random = False, show_card = True):
        self.name = name
        self.cards = []
        self.hand_rank = 0
        self.best_hand = {
            'type':PokerHand.HighestCard,
            'kickers':[]
        }
        self.wallet = wallet
        self.bet = 0
        self.in_game = True
        self.is_all_in = False
        self.role = Roles.Player
        self.random = random
        self.show_card = show_card
    
    def show_cards(self):
        return [str(card) for card in self.cards]
    
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
        self.big_blind = 2
        self.community_cards = []
        self.is_done = False
        self.is_pre_flop = True
        self.pot = self.small_blind + self.big_blind
        self.winners = []
        self.round_num = 0
        self.verbose = verbose
        #self.raise_history = [1,0,0,1,0,0]

    def get_actions(self):
        actions = []
        if self.active_player.bet == self.round.bet_to_call:
            actions.append(Actions.Check)
            actions.append(Actions.Raise)
        elif self.active_player.bet < self.round.bet_to_call:
            actions.append(Actions.Call)
            if (self.active_player.wallet + self.active_player.bet) > self.round.bet_to_call and self.round.num_raises < 4:
                actions.append(Actions.Raise)
            actions.append(Actions.Fold)
        return [action.value for action in actions]

    
    def deal_cards(self, num = 2):
        for player in self.players:
            for _ in range(num):
                player.cards.append(self.deck.deal_card())
    
    def show_card(self, num = 1):
        for _ in range(num):
            card = self.deck.deal_card()
            self.community_cards.append(card)
            #print(card)
    
    def new_round(self, bet_to_call = 0):
        self.round_num += 1
        self.round = Round(bet_to_call)
        if self.verbose:
            print(f"\nRound {self.round_num}")
    
    def reset(self):
        if self.verbose:
            print("\nNew game!")
        self.deck = Deck()
        self.active_player = None
        self.community_cards = []
        self.is_done = False
        self.is_pre_flop = True
        self.pot = self.small_blind + self.big_blind
        self.winners = []
        self.round_num = 0
        #self.raise_history = [1,0,0,1,0,0]

        for player in self.players:
            player.wallet = 50
            player.cards = []
            player.best_hand = {
                'type':PokerHand.HighestCard,
                'kickers':[]
            }

        self.deal_cards()

        self.new_round(self.big_blind)
        if random.randint(0, 1):
            self.active_player = self.players[1]
            self.players[0].role = Roles.SmallBlind
            self.players[1].role = Roles.BigBlind
        else:
            self.active_player = self.players[0]
            self.players[0].role = Roles.BigBlind
            self.players[1].role = Roles.SmallBlind
        
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
                    print("Error, wallet size is to small for big blind")

        if self.verbose:
            self.print_info()
    
    def do_action(self, action):
        if self.verbose:
            print(f"{Actions(action).name}!")
        valid_actions = self.get_actions()
        if action in valid_actions:

            if action == Actions.Check.value:
                self.round.last_action = Actions.Check.value

            elif action == Actions.Call.value:
                bet = self.round.bet_to_call - self.active_player.bet
                self.active_player.wallet -= bet
                self.active_player.bet += bet
                self.pot += bet
                self.round.last_action = Actions.Call.value

            elif action == Actions.Raise.value:
                bet_size = self.small_blind
                if self.round_num>2:
                    bet_size = self.big_blind
                self.round.bet_to_call += bet_size
                self.round.num_raises += 1
                bet = self.round.bet_to_call - self.active_player.bet
                self.active_player.wallet -= bet
                self.active_player.bet += bet
                self.pot += bet
                self.round.last_action = Actions.Raise.value
                #self.raise_history[3*(self.round_num-1):3*self.round_num] = [0]*3
                #self.raise_history[self.round.num_raises+3*(self.round_num-1)] = 1

            elif action == Actions.Fold.value:
                self.is_done = True
                self.round.last_action = Actions.Fold.value
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
                
                if self.round_num == 1:
                    self.show_card(3)
                    self.new_round()
                elif self.round_num == 2:
                    self.show_card()
                    self.new_round()
                elif self.round_num == 3:
                    self.show_card()
                    self.new_round()
                else:
                    self.is_done = True
                    self.get_winner()

        self.active_player = [player for player in self.players if player != self.active_player][0]
        
        if self.verbose and not self.is_done:
            self.print_info()
    
    def get_winner(self):
        if not self.winners:
            for player in self.players:
                all_cards = player.cards + self.community_cards
                ranks = sorted([card.rank.value for card in all_cards], reverse=True)
                suits = [card.suit.value for card in all_cards]
                higest_card_rank = max(ranks)
                player.best_hand['type'] = PokerHand.HighestCard
                player.best_hand['kickers'] = [higest_card_rank]+[rank for rank in ranks if rank != higest_card_rank][:4]

                pairs = list(set([rank for rank in ranks if ranks.count(rank) == 2]))

                pair_rank = False
                if pairs:
                    pair_rank = max(pairs)
                    player.best_hand['type'] = PokerHand.Pair
                    player.best_hand['kickers'] = [pair_rank]+[rank for rank in ranks if rank != pair_rank][:3]

                if len(pairs)>1:
                    pairs.remove(max(pairs))
                    two_pair_rank = [pair_rank, max(pairs)]
                    player.best_hand['type'] = PokerHand.TwoPair
                    player.best_hand['kickers'] = two_pair_rank+[rank for rank in ranks if rank not in two_pair_rank][:1]

                threes = [rank for rank in ranks if ranks.count(rank) == 3]
                three_of_a_kind_rank = False
                if threes:
                    three_of_a_kind_rank = max(threes)
                    player.best_hand['type'] = PokerHand.ThreeOfAKind
                    player.best_hand['kickers'] = [three_of_a_kind_rank]+[rank for rank in ranks if rank != three_of_a_kind_rank][:2]

                straight_count = 1
                last_rank = -1
                unique_ranks = list(set(ranks))
                unique_ranks.sort()
                straight_rank = False
                for rank in unique_ranks:
                    if last_rank + 1 == rank:
                        straight_count += 1
                        if straight_count >= 5:
                            straight_rank = rank
                    else:
                        straight_count = 1
                    last_rank = rank
                if straight_rank:
                    player.best_hand['type'] = PokerHand.Straight
                    player.best_hand['kickers'] = [straight_rank]

                flush_suit_value = False
                flush_rank = False
                for suit in Suits:
                    if suits.count(suit.value) >= 5:
                        flush_rank = max([a for a in suits if a == suit.value])
                        flush_suit_value = suit.value
                        player.best_hand['type'] = PokerHand.Flush
                        player.best_hand['kickers'] = [flush_rank]
                        break #It's only possible to get one flush per hand so we don't have to look further

                if three_of_a_kind_rank and pair_rank:
                    full_house_rank = [three_of_a_kind_rank, pair_rank]
                    player.best_hand['type'] = PokerHand.FullHouse
                    player.best_hand['kickers'] = full_house_rank

                fours = [rank for rank in ranks if ranks.count(rank) == 4]
                four_of_a_kind_rank = False
                if fours:
                    four_of_a_kind_rank = max(fours)
                    player.best_hand['type'] = PokerHand.FourOfAKind
                    player.best_hand['kickers'] = [four_of_a_kind_rank] #No need for kicker, two players can't have the same four of a kind

                straight_flush_rank = False
                if flush_rank:
                    unique_flush_suite_ranks = list(set([card.rank.value for card in all_cards if card.suit.value == flush_suit_value]))
                    unique_flush_suite_ranks.sort()
                    straight_flush_count = 1
                    last_rank = -1
                    for rank in unique_flush_suite_ranks:
                        if last_rank + 1 == rank:
                            straight_flush_count += 1
                            if straight_flush_count >= 5:
                                straight_flush_rank = rank
                        else:
                            straight_flush_count = 1
                        last_rank = rank

                    if straight_flush_rank:
                        player.best_hand['type'] = PokerHand.StraightFlush
                        player.best_hand['kickers'] = [straight_flush_rank]

                        if straight_flush_rank == Ranks.Ace.value:
                            player.best_hand['type'] = PokerHand.RoyalFlush

                if self.winners:
                    if player.best_hand["type"].value > self.winners[0].best_hand["type"].value:
                        self.winners = [player]
                    elif player.best_hand["type"].value == self.winners[0].best_hand["type"].value:
                        if player.best_hand["kickers"] > self.winners[0].best_hand["kickers"]:
                            self.winners = [player]
                        elif player.best_hand["kickers"] == self.winners[0].best_hand["kickers"]:
                            self.winners.append(player)
                else:
                    self.winners = [player]
                
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
            print(f'{self.players[0].name} cards: {"".join([str(card) for card in self.players[0].cards])}')
            print(f'{self.players[1].name} cards: {"".join([str(card) for card in self.players[1].cards])}')
            print(f"Community cards: {''.join([str(card) for card in self.community_cards])}")
            print(f"Winning hand type: {self.winners[0].best_hand['type'].name}")
    
    def get_state(self):
        reward = 0
        if self.is_done:
            reward = self.players[0].wallet - 50
        
        state = [0]*(52+52+4)
        state[self.players[0].cards[0].id] = 1
        state[self.players[0].cards[1].id] = 1
        if self.community_cards:
            for card in self.community_cards:
                state[card.id+52] = 1
        if self.round.last_action is not None:
            state[self.round.last_action+104] = 1
        #state[12:] = self.raise_history


        return reward, state, self.is_done
    
    def print_info(self):
        valid_actions = self.get_actions()
        if self.active_player.show_card:
            print(f'\n{self.active_player.name} - {self.active_player.cards[0]}{self.active_player.cards[1]} - Wallet: {self.active_player.wallet} - Pot: {self.pot}')
            print(f'Bet to call: {self.round.bet_to_call}')
            if self.community_cards:
                print(f"Community cards: {''.join([str(card) for card in self.community_cards])}")
            print("\n".join(map(lambda x: f"{x}: {Actions(x).name}", valid_actions)))
        else:
            print(f'\n{self.active_player.name} - Wallet: {self.active_player.wallet}')
    
def pretty_card(card):
    if card.suit == Suits.Spades:
        suit = '♠️'
    elif card.suit == Suits.Hearts:
        suit = '♥️'
    elif card.suit == Suits.Clubs:
        suit = '♣️'
    elif card.suit == Suits.Diamonds:
        suit = '♦️'

    rank = card.rank.value

    if rank == 11:
        rank = 'J'
    elif rank == 12:
        rank = 'Q'
    elif rank == 13:
        rank = 'K'
    elif rank == 1:
        rank = 'A'
    
    return f'{suit}{rank}'