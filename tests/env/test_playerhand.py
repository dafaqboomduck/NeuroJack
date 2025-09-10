# tests/env/test_playerhand.py

import pytest
from neurojack.env.playerhand import PlayerHand
from neurojack.env.card import Card # PlayerHand depends on Card

# --- Fixtures for common Card instances ---
@pytest.fixture
def ace_spades():
    return Card('A', 'S')

@pytest.fixture
def ten_hearts():
    return Card('10', 'H')

@pytest.fixture
def two_clubs():
    return Card('2', 'C')

@pytest.fixture
def five_diamonds():
    return Card('5', 'D')

# --- Fixtures for PlayerHand instances ---
@pytest.fixture
def empty_hand():
    """Returns an empty PlayerHand."""
    return PlayerHand()

@pytest.fixture
def initial_hand(ten_hearts, two_clubs):
    """Returns a PlayerHand with initial two cards (10H, 2C)."""
    return PlayerHand(cards=[ten_hearts, two_clubs])

# --- Test PlayerHand Initialization ---
def test_playerhand_initialization_empty():
    """Test that a PlayerHand can be initialized empty."""
    hand = PlayerHand()
    assert hand.cards == []
    assert not hand.stood
    assert not hand.double_down
    assert hand.reward == 0.0
    assert not hand.is_split_ace

def test_playerhand_initialization_with_cards(ten_hearts, two_clubs):
    """Test that a PlayerHand can be initialized with a list of cards."""
    hand = PlayerHand(cards=[ten_hearts, two_clubs])
    assert hand.cards == [ten_hearts, two_clubs]
    assert not hand.stood
    assert not hand.double_down
    assert hand.reward == 0.0
    assert not hand.is_split_ace

# --- Test add_card method ---
def test_add_card_to_empty_hand(empty_hand, ace_spades):
    """Test adding a card to an empty hand."""
    empty_hand.add_card(ace_spades)
    assert empty_hand.cards == [ace_spades]

def test_add_card_to_existing_hand(initial_hand, five_diamonds):
    """Test adding a card to a hand that already has cards."""
    initial_hand.add_card(five_diamonds)
    # Original cards (10H, 2C) + new card (5D)
    assert len(initial_hand.cards) == 3
    assert initial_hand.cards[2] == five_diamonds

# --- Test State Flags ---
def test_stood_flag(empty_hand):
    """Test setting and unsetting the stood flag."""
    assert not empty_hand.stood
    empty_hand.stood = True
    assert empty_hand.stood
    empty_hand.stood = False
    assert not empty_hand.stood

def test_double_down_flag(empty_hand):
    """Test setting and unsetting the double_down flag."""
    assert not empty_hand.double_down
    empty_hand.double_down = True
    assert empty_hand.double_down
    empty_hand.double_down = False
    assert not empty_hand.double_down

def test_is_split_ace_flag(empty_hand):
    """Test setting and unsetting the is_split_ace flag."""
    assert not empty_hand.is_split_ace
    empty_hand.is_split_ace = True
    assert empty_hand.is_split_ace
    empty_hand.is_split_ace = False
    assert not empty_hand.is_split_ace

def test_reward_assignment(empty_hand):
    """Test assigning reward to the hand."""
    assert empty_hand.reward == 0.0
    empty_hand.reward = 1.0
    assert empty_hand.reward == 1.0
    empty_hand.reward = -0.5
    assert empty_hand.reward == -0.5

# --- Test String Representation ---
def test_playerhand_str_representation(initial_hand):
    """Test the __str__ method of PlayerHand."""
    # Assuming initial_hand has 10H, 2C
    expected_str_part = "Cards: ['10H', '2C']"
    assert expected_str_part in str(initial_hand)
    assert "Stood: False" in str(initial_hand)
    assert "DD: False" in str(initial_hand)
    assert "Reward: 0.00" in str(initial_hand)

def test_playerhand_str_representation_with_state_changes(empty_hand, ace_spades, ten_hearts):
    """Test __str__ with various state changes."""
    empty_hand.add_card(ace_spades)
    empty_hand.add_card(ten_hearts) # Hand is AS, 10H (21)
    empty_hand.stood = True
    empty_hand.double_down = True # This state combination isn't typical but tests the string
    empty_hand.reward = 1.5
    empty_hand.is_split_ace = True

    expected_str_part = "Cards: ['AS', '10H']"
    assert expected_str_part in str(empty_hand)
    assert "Stood: True" in str(empty_hand)
    assert "DD: True" in str(empty_hand)
    assert "Reward: 1.50" in str(empty_hand)
    # is_split_ace is not part of __str__ in your current PlayerHand.py, so no assertion for it.
