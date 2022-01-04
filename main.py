from enum import Enum
import random

class Suits(Enum):
    Spades = 0
    Hearts = 1
    Clubs = 2
    Diamonds = 3

class Ranks(Enum):
    Ace = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Jack = 11
    Queen = 12
    King = 13


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
    cards = []

class Player:
    def __init__(self, name, hand, wallet):
        self.name = name
        self.hand = hand
        self.wallet = wallet
        self.bet = 0

class Round:
    pass

class Game:
    def __init__(self, deck, players):
        self.deck = deck
        self.players = players
    
    def deal_cards(self):
        for player in self.players:
            player.hand.cards.append(self.deck.deal_card())
        

player1 = Player("Bob", Hand(), 100)
player2 = Player("Alice", Hand(), 100)
deck = Deck()
game1 = Game(deck, [player1, player2])
game1.deal_cards()

print(player1.hand.cards[0])
