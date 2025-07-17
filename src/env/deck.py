import random
import logging
from typing import List, Optional
from src.env.card import Card
from src.env.values import RANKS, SUITS

# Configure logging for the deck module
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Deck:
    """Manages a deck of cards, including shuffling and dealing."""
    def __init__(self, num_decks: int = 6, seed: Optional[int] = None, reshuffle_threshold_pct: float = 0.25):
        """
        Initializes the Deck.

        Args:
            num_decks (int): The number of standard 52-card decks to use.
            seed (Optional[int]): Seed for the random number generator for reproducibility.
            reshuffle_threshold_pct (float): Percentage of cards remaining at which to reshuffle.
                                             e.g., 0.25 means reshuffle when 25% or less cards are left.
        """
        if not (1 <= num_decks <= 8): # Common range for blackjack
            logger.warning(f"Number of decks ({num_decks}) is outside common range (1-8).")
        if not (0.0 <= reshuffle_threshold_pct < 1.0):
            raise ValueError("Reshuffle threshold percentage must be between 0.0 and 1.0 (exclusive of 1.0).")

        self.num_decks = num_decks
        self.initial_num_cards = self.num_decks * 52
        self.reshuffle_threshold = int(self.initial_num_cards * reshuffle_threshold_pct)
        self.cards: List[Card] = []
        self.seed = seed
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self._create_deck()
        self.shuffle()
        logger.info(f"Deck initialized with {self.num_decks} decks. Reshuffle threshold: {self.reshuffle_threshold} cards.")

    def _create_deck(self) -> None:
        """Creates a fresh set of decks."""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in SUITS:
                for rank in RANKS:
                    self.cards.append(Card(rank, suit))
        logger.debug(f"Created {len(self.cards)} cards in the deck.")

    def shuffle(self) -> None:
        """Shuffles the deck."""
        self._rng.shuffle(self.cards)
        logger.info("Deck shuffled.")

    def deal_card(self) -> Card:
        """
        Deals a single card from the top of the deck.
        Reshuffles if the number of remaining cards falls below the threshold.
        """
        if len(self.cards) <= self.reshuffle_threshold:
            logger.info(f"Deck count ({len(self.cards)}) below reshuffle threshold ({self.reshuffle_threshold}). Reshuffling.")
            self._create_deck()
            self.shuffle()
        return self.cards.pop()

    def cards_remaining(self) -> int:
        """Returns the number of cards currently in the deck."""
        return len(self.cards)