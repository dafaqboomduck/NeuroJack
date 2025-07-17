from typing import List, Optional
from src.env.card import Card

class PlayerHand:
    """Represents a player's hand, including its cards and state during play."""
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards: List[Card] = cards if cards is not None else []
        self.stood: bool = False        # True if the player has chosen to stand on this hand
        self.double_down: bool = False  # True if the player has doubled down on this hand
        self.reward: float = 0           # Stores the individual reward for this hand
        self.is_split_ace: bool = False # True if this hand originated from splitting Aces

    def add_card(self, card: Card) -> None:
        """Adds a card to the hand."""
        self.cards.append(card)

    def __str__(self) -> str:
        """String representation of the hand."""
        card_strs = [str(card) for card in self.cards]
        return f"Cards: {card_strs}, Stood: {self.stood}, DD: {self.double_down}, Reward: {self.reward:.2f}"