from src.env.values import CARD_VALUES, HI_LO_COUNT_VALUES

class Card:
    """Represents a playing card."""
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = CARD_VALUES[rank]
        self.count_value = HI_LO_COUNT_VALUES[rank]

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return f"Card('{self.rank}', '{self.suit}')"
