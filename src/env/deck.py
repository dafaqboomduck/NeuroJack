import random
import logging
from typing import List, Optional
from src.env.card import Card
from src.env.values import RANKS, SUITS

# Configure logging for the deck module
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Deck:
    """
    Represents one or more combined decks of playing cards for Blackjack.

    This class manages:
    - Creating a specified number of standard decks.
    - Shuffling cards.
    - Dealing cards one at a time.
    - Automatically reshuffling when the remaining cards fall below a threshold.

    Attributes:
        num_decks (int): Number of 52-card decks in use.
        initial_num_cards (int): Total cards at the start.
        reshuffle_threshold (int): The card count at which the deck reshuffles.
        cards (List[Card]): The current list of cards in the deck.
        seed (Optional[int]): Random seed for reproducible shuffling.
    """

    def __init__(self, num_decks: int = 6, seed: Optional[int] = None, reshuffle_threshold_pct: float = 0.25):
        """
        Initializes a new Deck instance.

        Args:
            num_decks (int): Number of decks to include (default is 6, typical for casinos).
            seed (Optional[int]): Random seed for reproducible shuffling.
            reshuffle_threshold_pct (float): Fraction of cards remaining at which reshuffling occurs
                                             (e.g., 0.25 means reshuffle when 25% or fewer cards remain).

        Raises:
            ValueError: If reshuffle_threshold_pct is not in the range [0.0, 1.0).
        """
        if not (1 <= num_decks <= 8):  # Casino blackjack usually uses 1-8 decks
            logger.warning(f"Number of decks ({num_decks}) is outside common range (1-8).")
        if not (0.0 <= reshuffle_threshold_pct < 1.0):
            raise ValueError("Reshuffle threshold percentage must be between 0.0 and 1.0 (exclusive of 1.0).")

        # Deck configuration
        self.num_decks = num_decks
        self.initial_num_cards = self.num_decks * 52
        self.reshuffle_threshold = int(self.initial_num_cards * reshuffle_threshold_pct)
        self.cards: List[Card] = []
        self.seed = seed
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()

        # Initialize the deck
        self._create_deck()
        self.shuffle()
        logger.debug(f"Deck initialized with {self.num_decks} decks. Reshuffle threshold: {self.reshuffle_threshold} cards.")

    def _create_deck(self) -> None:
        """
        Creates a fresh deck (or multiple decks) of 52 cards each.
        This method is called during initialization and when reshuffling is needed.
        """
        self.cards = [
            Card(rank, suit)
            for _ in range(self.num_decks)
            for suit in SUITS
            for rank in RANKS
        ]
        logger.debug(f"Created {len(self.cards)} cards in the deck.")

    def shuffle(self) -> None:
        """
        Shuffles the deck using the internal random number generator.
        """
        self._rng.shuffle(self.cards)
        logger.debug("Deck shuffled.")

    def deal_card(self) -> Card:
        """
        Deals (removes and returns) the top card from the deck.

        If the number of remaining cards falls below the reshuffle threshold,
        the deck is automatically recreated and shuffled.

        Returns:
            Card: The card dealt from the deck.
        """
        if len(self.cards) <= self.reshuffle_threshold:
            logger.debug(f"Deck count ({len(self.cards)}) below reshuffle threshold ({self.reshuffle_threshold}). Reshuffling.")
            self._create_deck()
            self.shuffle()
        return self.cards.pop()

    def cards_remaining(self) -> int:
        """
        Gets the current number of cards left in the deck.

        Returns:
            int: The count of remaining cards.
        """
        return len(self.cards)
