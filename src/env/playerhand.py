class PlayerHand:
    """Represents a player's hand, including its cards and state during play."""
    def __init__(self, cards=None):
        self.cards = cards if cards is not None else []
        self.stood = False        # True if the player has chosen to stand on this hand
        self.double_down = False  # True if the player has doubled down on this hand
        self.reward = 0           # Stores the individual reward for this hand

    def add_card(self, card):
        """Adds a card to the hand."""
        self.cards.append(card)

    def __str__(self):
        """String representation of the hand."""
        return f"Cards: {[str(card) for card in self.cards]}, Stood: {self.stood}, DD: {self.double_down}"
