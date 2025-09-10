# tests/env/test_deck.py

import pytest
from neurojack.env.deck import Deck
from neurojack.env.card import Card
from neurojack.env.values import RANKS, SUITS
import random # Import standard random for comparison if needed

# --- Fixtures for Deck instances ---
@pytest.fixture
def single_deck_no_shuffle_threshold():
    """Returns a single deck with reshuffle_threshold_pct = 0."""
    return Deck(num_decks=1, reshuffle_threshold_pct=0.0)

@pytest.fixture
def six_decks_standard_threshold():
    """Returns a standard 6-deck shoe with 25% reshuffle threshold."""
    return Deck(num_decks=6, reshuffle_threshold_pct=0.25)

@pytest.fixture
def seeded_deck():
    """Returns a single deck with a fixed seed for reproducibility."""
    return Deck(num_decks=1, seed=123, reshuffle_threshold_pct=0.25)

@pytest.fixture
def another_seeded_deck():
    """Returns a single deck with a different fixed seed for comparison."""
    return Deck(num_decks=1, seed=456, reshuffle_threshold_pct=0.25)

# --- Test Deck Initialization ---
def test_deck_initialization_defaults():
    """Test Deck initializes with default 6 decks and correct card count."""
    deck = Deck()
    assert deck.num_decks == 6
    assert deck.initial_num_cards == 6 * 52
    assert len(deck.cards) == 6 * 52 # Should be full after init and shuffle
    assert deck.reshuffle_threshold == int(6 * 52 * 0.25) # Default 25%

def test_deck_initialization_custom_num_decks():
    """Test Deck initializes with a custom number of decks."""
    deck = Deck(num_decks=2)
    assert deck.num_decks == 2
    assert deck.initial_num_cards == 2 * 52
    assert len(deck.cards) == 2 * 52

def test_deck_initialization_invalid_reshuffle_threshold_pct():
    """Test that invalid reshuffle_threshold_pct raises ValueError."""
    with pytest.raises(ValueError, match="Reshuffle threshold percentage must be between 0.0 and 1.0"):
        Deck(reshuffle_threshold_pct=1.0)
    with pytest.raises(ValueError, match="Reshuffle threshold percentage must be between 0.0 and 1.0"):
        Deck(reshuffle_threshold_pct=-0.1)

def test_deck_initialization_seed_creates_rng():
    """Test that providing a seed creates an internal random.Random instance."""
    deck = Deck(seed=1)
    assert isinstance(deck._rng, random.Random)
    # Cannot directly test if it's the *correct* seeded instance without exposing more internals,
    # but subsequent shuffle/deal tests will implicitly verify this.

# --- Test Shuffling ---
def test_deck_shuffle_randomness():
    """Test that shuffling changes the order of cards."""
    deck1 = Deck(num_decks=1, seed=None) # No seed, truly random
    deck2 = Deck(num_decks=1, seed=None) # No seed, truly random
    original_order1 = list(deck1.cards)
    original_order2 = list(deck2.cards) # Should be same as original_order1 before shuffle
    
    # After initialization, decks are already shuffled.
    # To test shuffling specifically, we need to create a fresh, unshuffled deck first.
    # This might require a small helper method or directly manipulating deck.cards in a test.
    # For now, we'll test reproducibility of seeded shuffles.
    
    # The default __init__ already shuffles, so we test that subsequent shuffles work.
    deck1._create_deck() # Reset to unshuffled state
    deck1.shuffle()
    assert original_order1 != deck1.cards # Should be different after shuffle

def test_deck_shuffle_reproducibility_with_seed(seeded_deck, another_seeded_deck):
    """Test that shuffling with the same seed produces the same order."""
    # seeded_deck is already initialized and shuffled once with seed 123
    # Create another deck with the same seed
    deck_same_seed = Deck(num_decks=1, seed=123, reshuffle_threshold_pct=0.25)
    
    # Compare their initial shuffled states
    assert seeded_deck.cards == deck_same_seed.cards

    # Shuffle both again and compare
    seeded_deck.shuffle()
    deck_same_seed.shuffle()
    assert seeded_deck.cards == deck_same_seed.cards

    # Ensure different seeds produce different orders
    assert seeded_deck.cards != another_seeded_deck.cards

# --- Test Dealing Cards ---
def test_deal_card_removes_from_top(single_deck_no_shuffle_threshold):
    """Test that deal_card removes the last card (top of stack)."""
    original_len = len(single_deck_no_shuffle_threshold.cards)
    top_card_before_deal = single_deck_no_shuffle_threshold.cards[-1]
    dealt_card = single_deck_no_shuffle_threshold.deal_card()
    assert dealt_card == top_card_before_deal
    assert len(single_deck_no_shuffle_threshold.cards) == original_len - 1

def test_cards_remaining(single_deck_no_shuffle_threshold):
    """Test cards_remaining method."""
    initial_cards = len(single_deck_no_shuffle_threshold.cards)
    assert single_deck_no_shuffle_threshold.cards_remaining() == initial_cards
    single_deck_no_shuffle_threshold.deal_card()
    assert single_deck_no_shuffle_threshold.cards_remaining() == initial_cards - 1

# --- Test Reshuffling Logic ---
def test_deck_auto_reshuffle_on_threshold():
    """Test that the deck automatically reshuffles when below threshold."""
    deck = Deck(num_decks=1, reshuffle_threshold_pct=0.1) # 52 * 0.1 = 5 cards threshold
    initial_num_cards = len(deck.cards)
    
    # Deal cards until just above threshold
    for _ in range(initial_num_cards - 5): # Deal 52 - 5 = 47 cards
        deck.deal_card()
    
    # Deck should have 5 cards left, not yet reshuffled
    assert len(deck.cards) == 5
    
    # Deal one more card, triggering reshuffle
    dealt_card = deck.deal_card() # This should trigger reshuffle
    
    # After reshuffle, the deck should be full again (52 cards)
    # FIX: The deck will have initial_num_cards - 1 because one card was just dealt from it.
    assert len(deck.cards) == deck.initial_num_cards - 1 # Should be a full new deck minus the dealt card
    assert deck.cards_remaining() == deck.initial_num_cards - 1 # And one card was dealt from it
    
    # Check that the dealt card is valid (not from the new deck, but the last of the old)
    # This is tricky to assert without knowing the exact order, but we can check type
    assert isinstance(dealt_card, Card)

def test_deck_reshuffle_maintains_seed_determinism():
    """
    Test that reshuffling in a seeded deck maintains determinism
    (i.e., the new shuffled order is predictable if seed is fixed).
    """
    deck1 = Deck(num_decks=1, seed=789, reshuffle_threshold_pct=0.1)
    deck2 = Deck(num_decks=1, seed=789, reshuffle_threshold_pct=0.1)

    # Deal cards from both until reshuffle is triggered
    initial_cards = len(deck1.cards)
    for _ in range(initial_cards - 5):
        deck1.deal_card()
        deck2.deal_card()

    # Trigger reshuffle for both
    deck1.deal_card()
    deck2.deal_card()

    # After reshuffle, the card order should be identical for both decks
    assert deck1.cards == deck2.cards
    # FIX: The deck will have initial_num_cards - 1 because one card was just dealt from it.
    assert len(deck1.cards) == deck1.initial_num_cards - 1
    assert len(deck2.cards) == deck2.initial_num_cards - 1
