from typing import List, Optional
from src.env.card import Card

class PlayerHand:
    """
    Represents a player's hand in Blackjack.

    This class stores:
    - The list of cards in the hand.
    - State flags indicating player decisions (stand, double down, split aces).
    - The reward associated with the outcome of this hand.

    Attributes:
        cards (List[Card]): The list of cards currently in the hand.
        stood (bool): Whether the player has chosen to stand on this hand.
        double_down (bool): Whether the player has doubled down on this hand.
        reward (float): The individual reward (positive, negative, or zero) assigned to this hand.
        is_split_ace (bool): Whether this hand originated from splitting Aces.
    """

    def __init__(self, cards: Optional[List[Card]] = None):
        """
        Initializes a PlayerHand.

        Args:
            cards (Optional[List[Card]]): Optional initial list of cards. Defaults to an empty list.
        """
        self.cards: List[Card] = cards if cards is not None else []
        self.stood: bool = False         # True if player chose to stand on this hand
        self.double_down: bool = False   # True if the player doubled down
        self.reward: float = 0           # Stores this hand's calculated reward
        self.is_split_ace: bool = False  # True if created by splitting Aces

    def add_card(self, card: Card) -> None:
        """
        Adds a card to the hand.

        Args:
            card (Card): The card to be added.
        """
        self.cards.append(card)

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the hand.

        Returns:
            str: A formatted string listing the cards and the hand's current state.
        """
        card_strs = [str(card) for card in self.cards]
        return f"Cards: {card_strs}, Stood: {self.stood}, DD: {self.double_down}, Reward: {self.reward:.2f}"
