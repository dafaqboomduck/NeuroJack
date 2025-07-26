# tests/env/test_card.py

import pytest
from src.env.card import Card
from src.env.values import CARD_VALUES, HI_LO_COUNT_VALUES

# --- Fixtures for common Card instances ---
@pytest.fixture
def ace_spades():
    """Returns an Ace of Spades card."""
    return Card('A', 'S')

@pytest.fixture
def ten_hearts():
    """Returns a Ten of Hearts card."""
    return Card('10', 'H')

@pytest.fixture
def two_clubs():
    """Returns a Two of Clubs card."""
    return Card('2', 'C')

# --- Test Card Initialization ---
def test_card_initialization_valid():
    """Test that a card can be initialized with valid rank and suit."""
    card = Card('K', 'D')
    assert card.rank == 'K'
    assert card.suit == 'D'
    assert card.value == 10
    assert card.count_value == -1

def test_card_initialization_ace():
    """Test Ace card values."""
    card = Card('A', 'H')
    assert card.rank == 'A'
    assert card.suit == 'H'
    assert card.value == 11
    assert card.count_value == -1

def test_card_initialization_low_card():
    """Test a low card (e.g., '5') values."""
    card = Card('5', 'C')
    assert card.rank == '5'
    assert card.suit == 'C'
    assert card.value == 5
    assert card.count_value == 1

def test_card_initialization_invalid_rank():
    """Test that initialization with an invalid rank raises ValueError."""
    with pytest.raises(ValueError, match="Invalid card rank"):
        Card('Invalid', 'S')

def test_card_initialization_invalid_suit():
    """Test that initialization with an invalid suit raises ValueError."""
    with pytest.raises(ValueError, match="Invalid card suit"):
        Card('K', 'X')

# --- Test Card Properties (value, count_value) ---
def test_card_value_property():
    """Test that card.value correctly reflects Blackjack value."""
    assert Card('K', 'S').value == 10
    assert Card('A', 'D').value == 11
    assert Card('7', 'C').value == 7

def test_card_count_value_property():
    """Test that card.count_value correctly reflects Hi-Lo count."""
    assert Card('2', 'S').count_value == 1
    assert Card('8', 'D').count_value == 0
    assert Card('K', 'C').count_value == -1

# --- Test String Representations ---
def test_card_str_representation(ace_spades, ten_hearts):
    """Test the __str__ method."""
    assert str(ace_spades) == 'AS'
    assert str(ten_hearts) == '10H'

def test_card_repr_representation(two_clubs):
    """Test the __repr__ method."""
    assert repr(two_clubs) == "Card('2', 'C')"

# --- Test Equality and Hashing ---
def test_card_equality():
    """Test that two cards are equal if they have the same rank and suit."""
    card1 = Card('Q', 'D')
    card2 = Card('Q', 'D')
    card3 = Card('Q', 'H')
    assert card1 == card2
    assert card1 != card3

def test_card_equality_with_non_card_object():
    """Test equality comparison with a non-Card object."""
    card = Card('J', 'C')
    assert (card == "JC") is False # Should not raise error, but return False

def test_card_hashing():
    """Test that Card objects are hashable and equal cards have equal hashes."""
    card1 = Card('J', 'C')
    card2 = Card('J', 'C')
    card3 = Card('J', 'D')
    assert hash(card1) == hash(card2)
    assert hash(card1) != hash(card3)
    # Test use in a set
    card_set = {card1, card3}
    assert len(card_set) == 2
    card_set.add(card2)
    assert len(card_set) == 2 # Adding duplicate should not increase size
