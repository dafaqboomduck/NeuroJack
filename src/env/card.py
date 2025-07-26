from src.env.values import CARD_VALUES, HI_LO_COUNT_VALUES

class Card:
    """
    Represents a single playing card used in Blackjack.

    Attributes:
        rank (str): The rank of the card (e.g., '2', 'J', 'A').
        suit (str): The suit of the card ('H', 'D', 'C', 'S').
        value (int): The Blackjack value of the card (e.g., 'K' = 10, 'A' = 11).
        count_value (int): The Hi-Lo count value for card counting strategies.
    """

    def __init__(self, rank: str, suit: str):
        """
        Initializes a Card instance.

        Args:
            rank (str): Card rank. Must be a key in CARD_VALUES.
            suit (str): Card suit. Must be one of ['H', 'D', 'C', 'S'].
            
        Raises:
            ValueError: If the provided rank or suit is invalid.
        """
        if rank not in CARD_VALUES:
            raise ValueError(f"Invalid card rank: {rank}")
        if suit not in ['H', 'D', 'C', 'S']:
            raise ValueError(f"Invalid card suit: {suit}")

        # Assign properties based on rank and suit
        self.rank = rank
        self.suit = suit
        self.value = CARD_VALUES[rank]             # Blackjack value
        self.count_value = HI_LO_COUNT_VALUES[rank] # Hi-Lo counting value

    # Returns a user-friendly string representation of the card.
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    # Returns an unambiguous string representation of the card.
    def __repr__(self) -> str:
        return f"Card('{self.rank}', '{self.suit}')"
    
    # Checks if two cards are equal (same rank and suit).
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit
    
    # Allows Card objects to be used in sets and as dictionary keys.
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
