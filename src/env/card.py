from src.env.values import CARD_VALUES, HI_LO_COUNT_VALUES

class Card:
    """Represents a playing card."""
    def __init__(self, rank: str, suit: str):
        if rank not in CARD_VALUES:
            raise ValueError(f"Invalid card rank: {rank}")
        if suit not in ['H', 'D', 'C', 'S']:
            raise ValueError(f"Invalid card suit: {suit}")

        self.rank = rank
        self.suit = suit
        self.value = CARD_VALUES[rank]
        self.count_value = HI_LO_COUNT_VALUES[rank]

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card('{self.rank}', '{self.suit}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))