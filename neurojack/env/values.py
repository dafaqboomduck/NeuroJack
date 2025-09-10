# Card values for standard deck
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}
"""
CARD_VALUES (dict): Maps card ranks to their numeric values used in Blackjack.
- Number cards ('2'-'10') have their face values.
- Face cards ('J', 'Q', 'K') are worth 10.
- Ace ('A') is worth 11 by default (can also be treated as 1 in game logic).
"""

# Hi-Lo card counting values
HI_LO_COUNT_VALUES = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}
"""
HI_LO_COUNT_VALUES (dict): Maps card ranks to their Hi-Lo counting values.
Used in card counting strategies:
- Low cards ('2'-'6') increase the count (+1).
- Neutral cards ('7'-'9') do not affect the count (0).
- High cards ('10', 'J', 'Q', 'K', 'A') decrease the count (-1).
"""

# Suits and Ranks for deck creation
RANKS = list(CARD_VALUES.keys())
"""RANKS (list): Contains all card ranks used to generate a deck."""

SUITS = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
"""SUITS (list): Represents the four suits in a standard deck of cards."""
