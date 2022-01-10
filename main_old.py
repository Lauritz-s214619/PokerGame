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
        self.best_hand = {
            'type':PokerHand.HighestCard,
            'kickers':[]
        }
        self.wallet = wallet
        self.bet = 0
        self.in_game = True
        self.is_all_in = False
        self.role = Roles.Player
    
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
        self.active_players = players.copy()
        self.small_blind = 5
        self.big_blind = 10
        self.community_cards = []
        self.is_done = False
        self.is_pre_flop = True
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
        cycle_active_players = cycle(self.players)
        num_turns = 0

        #Align the iter to one player before the player who starts
        if self.is_pre_flop:
            #The person after big blind starts
            player = next(player for player in cycle_active_players if player.role == Roles.BigBlind)
            #Add blinds to pot
            round.pot = self.small_blind + self.big_blind
            round.bet_to_call = self.big_blind
        else:
            #The person after button starts
            player = next(player for player in cycle_active_players if player.role == Roles.Button)
        
        while(not round.is_done):
            num_active_players = len(self.active_players)
            if num_active_players <= 1:
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
                if bets.count(bets[0]) == len(bets) and num_turns >= num_active_players:
                    round.is_done = True

        round.pot = sum(bets)
        for player in self.players:
            player.bet = 0

 
player1 = Player("Bob", Hand(), 100)
player2 = Player("Alice", Hand(), 100)
player3 = Player("Anton", Hand(), 100)
player4 = Player("Lasse", Hand(), 100)

player1.role = Roles.Button
player2.role = Roles.SmallBlind
player3.role = Roles.BigBlind
game1 = Game(Deck(), [player1, player2,player3, player4])
game1.deal_cards(2)
# print(f'''
# =============================
# {player1.name}: {player1.show_cards()}
# {player2.name}: {player2.show_cards()}
# =============================
# ''')

print("Round 1:")
game1.new_round()

print("\nRound 2 - Flop:")
game1.show_card(3)
game1.is_pre_flop = False
game1.new_round()

print("\nRound 3 - Turn:")
game1.show_card()

game1.new_round()

print("\nRound 4 - River:")
game1.show_card()

game1.new_round()

#game1.get_winner()
#print("\n")

winning_players = []
for player in game1.active_players:
    all_cards = player.hand.cards + game1.community_cards

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

    if winning_players:
        if player.best_hand["type"].value > winning_players[0].best_hand["type"].value:
            winning_players = [player]
        elif player.best_hand["type"].value == winning_players[0].best_hand["type"].value:
            if player.best_hand["kickers"] > winning_players[0].best_hand["kickers"]:
                winning_players = [player]
            elif player.best_hand["kickers"] == winning_players[0].best_hand["kickers"]:
                winning_players.append(player)
    else:
        winning_players = [player]

if len(winning_players)>1:
    print("Tie!")
for player in winning_players:
    print(f'{player.name} wins! ({player.best_hand["type"].name})')