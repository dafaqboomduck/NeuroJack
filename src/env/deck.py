from src.env.values import CARD_VALUES
from src.env.card import Card
import random

class Deck:
    """Manages a deck of cards, including shuffling and dealing."""
    def __init__(self, num_decks=1, seed=None):
        self.num_decks = num_decks
        self.cards = []
        self._create_deck()
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.shuffle()

    def _create_deck(self):
        """Creards = [] # Clear existing cards if recreating
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(rank, suit))
…            print("Reshuffling deck as it ran out of cards.")
        return self.cards.pop()ates a fresh set of decks."""
        ranks = list(CARD_VALUES.keys())
        suits = ['H', 'D', 'C', 'S'] # Hearts, Diamonds, Clubs, Spades
        self.cards = [] # Clear existing cards if recreating
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(rank, suit))

    def shuffle(self):
        """Shuffles the deck."""
        random.shuffle(self.cards)

    def deal_card(self):
        """Deals a single card from the top of the deck. Reshuffles if empty."""
        if not self.cards:
            self._create_deck()
            self.shuffle()
            print("Reshuffling deck as it ran out of cards.")
        return self.cards.pop()