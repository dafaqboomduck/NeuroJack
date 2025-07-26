# tests/env/test_env.py

import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch

from src.env.env import CustomBlackjackEnv, ACTION_STAND, ACTION_HIT, ACTION_DOUBLE_DOWN, ACTION_SPLIT
from src.env.card import Card
from src.env.playerhand import PlayerHand
from src.env.deck import Deck # Import Deck to mock or inspect its behavior if needed

# Suppress environment's INFO/DEBUG logs during testing for cleaner output
# You can change this to logging.INFO or logging.DEBUG if you want to see env logs during test runs
logging.getLogger('src.env.env').setLevel(logging.CRITICAL)
logging.getLogger('src.env.deck').setLevel(logging.CRITICAL)

# --- Fixtures for common environment setups ---
@pytest.fixture
def basic_env():
    """Returns a basic environment with default rules, no card counting, no doubling/splitting."""
    return CustomBlackjackEnv(num_decks=1, blackjack_payout=1.5,
                              allow_doubling=False, allow_splitting=False,
                              count_cards=False)

@pytest.fixture
def advanced_env():
    """Returns an environment with doubling, splitting, and card counting enabled."""
    return CustomBlackjackEnv(num_decks=6, blackjack_payout=1.5,
                              allow_doubling=True, allow_splitting=True,
                              count_cards=True)

@pytest.fixture
def mock_deck():
    """
    A mock Deck to control card dealing for specific test scenarios.
    It deals cards in reverse order of the list provided to mock_cards.
    """
    m = MagicMock(spec=Deck)
    m.cards_remaining.return_value = 52 # Default, can be overridden
    m.running_count = 0 # Mock running count for testing purposes
    m.initial_num_cards = 52 # For true_count calculation if needed
    
    def mock_deal_card(mock_cards_list):
        # This closure captures the list of cards to deal
        def _deal():
            if not mock_cards_list:
                raise IndexError("Mock deck is empty!")
            card = mock_cards_list.pop(0) # Pop from the beginning to deal in order
            return card
        return _deal

    m.mock_deal_card_with_list = mock_deal_card # Attach helper to mock
    return m

# --- Helper for creating specific cards ---
def card(rank, suit):
    return Card(rank, suit)

# --- Test Environment Initialization ---
def test_env_initialization_basic(basic_env):
    """Test basic environment initialization."""
    assert basic_env.num_decks == 1
    assert basic_env.blackjack_payout == 1.5
    assert not basic_env.allow_doubling
    assert not basic_env.allow_splitting
    assert not basic_env.count_cards
    assert basic_env.state_size == 3
    assert basic_env.num_actions == 2 # Stand, Hit
    assert isinstance(basic_env.deck, Deck)
    assert len(basic_env.player_hands) == 1
    assert len(basic_env.dealer_hand) == 2 # Dealer gets 2 cards initially

def test_env_initialization_advanced(advanced_env):
    """Test advanced environment initialization."""
    assert advanced_env.num_decks == 6
    assert advanced_env.allow_doubling
    assert advanced_env.allow_splitting
    assert advanced_env.count_cards
    assert advanced_env.state_size == 5
    assert advanced_env.num_actions == 4 # Stand, Hit, Double Down, Split

    # Re-initialize advanced_env with a mock deck to control initial cards
    # This is a better way to test the initial running_count
    with patch('src.env.env.Deck') as MockDeckClass:
        mock_deck_instance = MockDeckClass.return_value
        
        # Scenario 1: Cards resulting in a running_count of 0
        mock_deck_instance.deal_card.side_effect = [
            card('7', 'H'),  # Player 1 (0 count)
            card('8', 'S'),  # Dealer Up (0 count)
            card('9', 'D'),  # Player 2 (0 count)
            card('K', 'C')   # Dealer Hole (-1 count)
        ]
        # Set cards_remaining for the mock deck.
        # Initial deal deals 4 cards. For 6 decks (312 cards total), 308 remain.
        mock_deck_instance.cards_remaining.return_value = (6 * 52) - 4
        
        mock_env = CustomBlackjackEnv(num_decks=6, blackjack_payout=1.5,
                                      allow_doubling=True, allow_splitting=True,
                                      count_cards=True)
        # Expected running count: 0 (7H) + 0 (8S) + 0 (9D) = 0
        assert mock_env.running_count == 0

        # Scenario 2: Cards resulting in a running_count of 3
        mock_deck_instance.deal_card.side_effect = [
            card('2', 'H'),  # Player 1 (+1 count)
            card('3', 'S'),  # Dealer Up (+1 count)
            card('4', 'D'),  # Player 2 (+1 count)
            card('K', 'C')   # Dealer Hole (-1 count)
        ]
        # Reset cards_remaining for the new scenario
        mock_deck_instance.cards_remaining.return_value = (6 * 52) - 4
        
        mock_env_2 = CustomBlackjackEnv(num_decks=6, blackjack_payout=1.5,
                                        allow_doubling=True, allow_splitting=True,
                                        count_cards=True)
        # Expected running count: 1 (2H) + 1 (3S) + 1 (4D) = 3
        assert mock_env_2.running_count == 3


# --- Test _update_hand_value helper method ---
def test_update_hand_value_no_ace():
    """Test hand value calculation without aces."""
    hand_cards = [card('10', 'H'), card('5', 'D')]
    s, ua = CustomBlackjackEnv._update_hand_value(None, hand_cards) # Pass None for self
    assert s == 15
    assert not ua

def test_update_hand_value_soft_ace():
    """Test hand value calculation with a usable ace."""
    hand_cards = [card('A', 'S'), card('6', 'D')]
    s, ua = CustomBlackjackEnv._update_hand_value(None, hand_cards)
    assert s == 17
    assert ua # Ace counted as 11

def test_update_hand_value_hard_ace():
    """Test hand value calculation where ace must be 1."""
    hand_cards = [card('A', 'S'), card('10', 'D'), card('5', 'C')]
    s, ua = CustomBlackjackEnv._update_hand_value(None, hand_cards)
    assert s == 16 # Ace as 1
    assert not ua

def test_update_hand_value_multiple_aces():
    """Test hand value calculation with multiple aces."""
    # Explicitly create cards to rule out fixture issues
    hand_cards = [Card('A', 'S'), Card('A', 'D'), Card('9', 'C')]
    s, ua = CustomBlackjackEnv._update_hand_value(None, hand_cards)
    assert s == 21 # One ace as 11, one as 1
    assert ua # One ace is still usable as 11

# --- Test _get_obs helper method ---
@patch('src.env.env.Deck') # Mock Deck to control dealt cards
def test_get_obs_no_card_counting(MockDeck): # Removed basic_env fixture, will create mocked env
    """Test observation without card counting."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('10', 'H'), # Player 1
        card('5', 'S'),  # Dealer Up
        card('7', 'D'),  # Player 2
        card('K', 'C')   # Dealer Hole
    ]
    # Set cards_remaining for the mock deck
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 4
    
    env = CustomBlackjackEnv(num_decks=1, count_cards=False) # Create env with mock
    # After reset, player has 10+7=17, dealer showing 5
    obs = env._get_obs()
    assert obs == (17, 5, 0) # player_sum, dealer_showing, usable_ace

@patch('src.env.env.Deck') # Mock Deck to control dealt cards
def test_get_obs_with_card_counting(MockDeck): # Removed advanced_env fixture, will create mocked env
    """Test observation with card counting."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.cards_remaining.return_value = 50 # Simulate 2 cards dealt for 1 deck
    mock_deck_instance.deal_card.side_effect = [
        card('2', 'H'),  # Player 1 (+1)
        card('10', 'S'), # Dealer Up (-1)
        card('3', 'D'),  # Player 2 (+1)
        card('J', 'C')   # Dealer Hole (-1, revealed later)
    ]
    env = CustomBlackjackEnv(num_decks=1, count_cards=True) # Use 1 deck for simpler true count
    # After reset, initial count: Player 2 (+1) + Player 3 (+1) + Dealer 10 (-1) = +1
    # Player has 2+3=5, dealer showing 10
    obs = env._get_obs()
    # Expected: player_sum=5, dealer_showing=10, usable_ace=0, running_count=1, true_count=1
    # true_count = running_count / (decks_remaining = 50/52) = 1.04, rounded to 1
    assert obs == (5, 10, 0, 1, 1)

# --- Test Reset Method ---
@patch('src.env.env.Deck') # Mock Deck to control dealt cards
def test_reset_initial_deal(MockDeck): # Removed basic_env fixture, will create mocked env
    """Test that reset deals correct number of cards and initializes state."""
    mock_deck_instance = MockDeck.return_value
    # Need 8 cards: 4 for __init__ reset, 4 for explicit reset call
    mock_deck_instance.deal_card.side_effect = [
        card('K', 'H'), card('A', 'S'), card('J', 'D'), card('2', 'C'), # For __init__ reset
        card('K', 'H'), card('A', 'S'), card('J', 'D'), card('2', 'C')  # For explicit reset
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 8
    # Re-initialize env to use the mock
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)

    # Explicitly call reset to test its behavior
    env.reset()

    assert len(env.player_hands) == 1
    assert len(env.player_hands[0].cards) == 2
    assert len(env.dealer_hand) == 2
    assert env.current_hand_index == 0
    assert env.running_count == 0 # No card counting
    
    # Verify cards dealt
    assert env.player_hands[0].cards[0] == card('K', 'H')
    assert env.dealer_hand[0] == card('A', 'S') # Dealer up-card
    assert env.player_hands[0].cards[1] == card('J', 'D')
    assert env.dealer_hand[1] == card('2', 'C') # Dealer hole-card

@patch('src.env.env.Deck')
def test_reset_player_blackjack(MockDeck):
    """Test reset handles player blackjack."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('A', 'H'), card('5', 'S'), card('10', 'D'), card('K', 'C')
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 4
    # Instantiate env, which calls reset internally
    env = CustomBlackjackEnv(num_decks=1, count_cards=False)
    
    assert env.player_hands[0].reward == 1.5 # Blackjack payout
    assert env.player_hands[0].stood # Hand should be stood if game is done
    assert env.current_hand_index == 0 # Index doesn't advance if game is done
    assert env.dealer_hand[1] == card('K', 'C') # Hole card should remain hidden in obs
    assert env._get_obs() == (21, 5, 1) # Player 21, Dealer showing 5, Usable Ace

@patch('src.env.env.Deck')
def test_reset_dealer_blackjack(MockDeck):
    """Test reset handles dealer blackjack."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('5', 'H'), card('A', 'S'), card('6', 'D'), card('10', 'C')
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 4
    env = CustomBlackjackEnv(num_decks=1, count_cards=False)
    
    assert env.player_hands[0].reward == -1.0
    assert env.player_hands[0].stood # Hand should be stood if game is done
    assert env.current_hand_index == 0
    assert env._get_obs() == (11, 11, 0) # Player 11, Dealer showing Ace, No usable ace

@patch('src.env.env.Deck')
def test_reset_both_blackjack_push(MockDeck):
    """Test reset handles both player and dealer blackjack (push)."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('A', 'H'), card('A', 'S'), card('10', 'D'), card('10', 'C')
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 4
    env = CustomBlackjackEnv(num_decks=1, count_cards=False)
    
    assert env.player_hands[0].reward == 0.0
    assert env.player_hands[0].stood
    assert env._get_obs() == (21, 11, 1)

@patch('src.env.env.Deck')
def test_reset_info_can_double_can_split(MockDeck):
    """Test reset info for doubling and splitting."""
    mock_deck_instance = MockDeck.return_value
    # Need 8 cards: 4 for __init__ reset, 4 for explicit reset call
    mock_deck_instance.deal_card.side_effect = [
        card('8', 'H'), card('5', 'S'), card('8', 'D'), card('K', 'C'), # For __init__ reset
        card('8', 'H'), card('5', 'S'), card('8', 'D'), card('K', 'C')  # For explicit reset
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 8
    
    # Instantiate env within the patch to ensure its __init__ (which calls reset)
    # uses the mocked deal_card and we capture the correct initial_info.
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=True, allow_splitting=True, count_cards=False)
    
    initial_obs, initial_info = env.reset() # Explicitly call reset to get info

    assert initial_info["can_double"] == True
    assert initial_info["can_split"] == True
    assert initial_obs == (16, 5, 0)

@patch('src.env.env.Deck')
def test_reset_info_no_double_no_split_if_disabled(MockDeck):
    """Test reset info when doubling/splitting are disabled."""
    mock_deck_instance = MockDeck.return_value
    # Need 8 cards: 4 for __init__ reset, 4 for explicit reset call
    mock_deck_instance.deal_card.side_effect = [
        card('8', 'H'), card('5', 'S'), card('8', 'D'), card('K', 'C'), # For __init__ reset
        card('8', 'H'), card('5', 'S'), card('8', 'D'), card('K', 'C')  # For explicit reset
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 8
    
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)

    initial_obs, initial_info = env.reset()

    assert initial_info["can_double"] == False
    assert initial_info["can_split"] == False
    assert initial_obs == (16, 5, 0)

# --- Test Step Method - Basic Actions (Hit, Stand) ---
@patch('src.env.env.Deck')
def test_step_player_hit_and_busts(MockDeck):
    """Test player hits and busts."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('10', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal: P=18, D=5 (hole 2)
        card('K', 'H'), # Player hits and busts (18+10=28)
        card('K', 'S')  # Dealer's hit card (5+2=7, hits K=17)
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 6 # 4 initial + 1 player hit + 1 dealer hit
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)
    # After reset: P=18, D=5
    
    # Player hits
    next_obs, reward, done, info = env.step(ACTION_HIT)
    
    assert done == True # Game should be done after bust
    assert reward == -1.0 # Player busts, immediate -1 reward
    assert env.player_hands[0].reward == -1.0 # Hand's individual reward
    assert env.player_hands[0].stood == True # Crucial fix: Hand should be stood
    assert env.current_hand_index == 1 # Should advance to next hand index, then resolve game
    assert next_obs == (28, 5, 0) # Observation reflects busted hand
    assert info == {"can_double": False, "can_split": False} # No actions available

@patch('src.env.env.Deck')
def test_step_player_hit_and_continues(MockDeck):
    """Test player hits and game continues."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('5', 'H'), card('5', 'S'), card('3', 'D'), card('2', 'C'), # Initial deal: P=8, D=5
        card('K', 'H') # Player hits (8+10=18)
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 5
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)
    # After reset: P=8, D=5

    # Player hits
    next_obs, reward, done, info = env.step(ACTION_HIT)
    
    assert done == False # Game should not be done
    assert reward == 0.0 # No immediate reward
    assert env.player_hands[0].reward == 0.0 # Hand reward not yet set
    assert env.player_hands[0].stood == False # Hand not stood
    assert env.current_hand_index == 0 # Still on current hand
    assert next_obs == (18, 5, 0) # Player sum 18, Dealer showing 5
    assert info == {"can_double": False, "can_split": False} # No double/split after first hit

@patch('src.env.env.Deck')
def test_step_player_stand_and_dealer_plays(MockDeck):
    """Test player stands and dealer plays out their hand."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('10', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal: P=18, D=5 (hole 2)
        card('K', 'H'), # Dealer hits (5+2=7 -> 7+10=17)
        card('3', 'C')  # Next card (not used by dealer)
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 5
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)
    # After reset: P=18, D=5 (hole 2) -> Dealer has 7
    
    # Player stands
    next_obs, reward, done, info = env.step(ACTION_STAND)
    
    assert done == True # Game should be done
    assert reward == 1.0 # P=18, D=17. Player wins.
    assert env.player_hands[0].reward == 1.0
    assert env.player_hands[0].stood == True
    assert env.current_hand_index == 1 # Should advance past player hands
    assert next_obs == (18, 5, 0) # Observation is still for player's hand state
    assert info == {"can_double": False, "can_split": False}

@patch('src.env.env.Deck')
def test_step_player_stand_and_dealer_busts(MockDeck):
    """Test player stands and dealer busts."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('10', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal: P=18, D=5 (hole 2)
        card('Q', 'H'), # Dealer hits (5+2=7 -> 7+10=17) -> (5+2+10+10=27) - BUST
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 6
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)
    # After reset: P=18, D=5 (hole 2) -> Dealer has 7
    
    # Player stands
    next_obs, reward, done, info = env.step(ACTION_STAND)
    
    assert done == True # Game should be done
    assert reward == 1.0 # P=18, D=busts. Player wins.
    assert env.player_hands[0].reward == 1.0
    assert env.player_hands[0].stood == True
    assert env.current_hand_index == 1
    assert next_obs == (18, 5, 0)
    assert info == {"can_double": False, "can_split": False}

# --- Test Step Method - Double Down ---
@patch('src.env.env.Deck')
def test_step_player_double_down_win(MockDeck):
    """Test player doubles down and wins."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('5', 'H'), card('5', 'S'), card('4', 'D'), card('2', 'C'), # Initial deal: P=9, D=5 (hole 2)
        card('K', 'H'), # Player double down card (9+10=19)
        card('K', 'S'), # Dealer hits (5+2+10=17)
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 6
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=True, allow_splitting=False, count_cards=False)
    # After reset: P=9, D=5 (hole 2)
    
    # Player doubles down
    next_obs, reward, done, info = env.step(ACTION_DOUBLE_DOWN)
    
    assert done == True
    assert reward == 2.0 # Doubled reward for win
    assert env.player_hands[0].reward == 2.0
    assert env.player_hands[0].double_down == True
    assert env.player_hands[0].stood == True
    assert len(env.player_hands[0].cards) == 3 # Should have 3 cards
    assert env.player_hands[0].cards[2] == card('K', 'H')
    assert next_obs == (19, 5, 0) # Player sum 19, Dealer showing 5

@patch('src.env.env.Deck')
def test_step_player_double_down_bust(MockDeck):
    """Test player doubles down and busts."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('10', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal: P=18, D=5 (hole 2)
        card('5', 'H'), # Player double down card (18+5=23) - BUST
        card('A', 'S')  # Add enough cards for dealer to play out, even if player busts.
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 6 # Adjusted for one more dealer card
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=True, allow_splitting=False, count_cards=False)
    # After reset: P=18, D=5 (hole 2)
    
    # Player doubles down
    next_obs, reward, done, info = env.step(ACTION_DOUBLE_DOWN)
    
    assert done == True
    assert reward == -2.0 # Doubled penalty for bust
    assert env.player_hands[0].reward == -2.0
    assert env.player_hands[0].double_down == True
    assert env.player_hands[0].stood == True
    assert len(env.player_hands[0].cards) == 3
    assert env.player_hands[0].cards[2] == card('5', 'H')
    assert next_obs == (23, 5, 0)

# --- Test Step Method - Split ---
@patch('src.env.env.Deck')
def test_step_player_split_non_aces(MockDeck):
    """Test player splits a pair (non-aces) and plays both hands."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('8', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal: P=8,8 (16), D=5 (hole 2)
        card('3', 'H'), # Card for first split hand (8+3=11)
        card('K', 'D'), # Card for second split hand (8+10=18)
        card('J', 'C'), # Player hits first hand (11+10=21)
        card('Q', 'S'), # Dealer hits (5+2+10=17)
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 7 # 4 initial + 2 split + 1 hit by player
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=True, allow_splitting=True, count_cards=False)
    # After reset: P=[8H, 8D], D=[5S, 2C]
    
    # Player splits
    next_obs, reward, done, info = env.step(ACTION_SPLIT)
    
    assert done == False # Game not done, player has more hands to play
    assert reward == 0.0 # No immediate reward
    assert len(env.player_hands) == 2 # Should now have two hands
    assert env.player_hands[0].cards == [card('8', 'H'), card('3', 'H')] # First hand
    assert env.player_hands[1].cards == [card('8', 'D'), card('K', 'D')] # Second hand
    assert env.player_hands[0].is_split_ace == False
    assert env.player_hands[1].is_split_ace == False
    assert env.current_hand_index == 0 # Still on the first hand
    assert next_obs == (11, 5, 0) # Observation for the first split hand (8+3=11)
    assert info["can_double"] == True # Can double on 11
    assert info["can_split"] == False # Cannot split after split

    # Player hits on first hand (11 + J = 21)
    next_obs, reward, done, info = env.step(ACTION_HIT)
    assert done == False
    assert reward == 0.0
    assert env.player_hands[0].cards == [card('8', 'H'), card('3', 'H'), card('J', 'C')]
    assert env.player_hands[0].stood == False # Not stood yet

    # Player stands on first hand (21)
    next_obs, reward, done, info = env.step(ACTION_STAND)
    assert done == False # Still second hand to play
    assert reward == 0.0
    assert env.player_hands[0].stood == True
    assert env.current_hand_index == 1 # Advanced to second hand
    assert next_obs == (18, 5, 0) # Observation for second hand (8+10=18)

    # Player stands on second hand (18)
    next_obs, reward, done, info = env.step(ACTION_STAND)
    assert done == True # All player hands resolved, dealer plays
    # Dealer has 5+2=7, hits with Q (17) -> stands.
    # Hand 1 (21) vs Dealer (17) -> Win (1.0)
    # Hand 2 (18) vs Dealer (17) -> Win (1.0)
    assert reward == 2.0 # Sum of rewards for both hands
    assert env.player_hands[0].reward == 1.0
    assert env.player_hands[1].reward == 1.0

@patch('src.env.env.Deck')
def test_step_player_split_aces(MockDeck):
    """Test player splits aces (auto-stands after one card)."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('A', 'H'), card('5', 'S'), card('A', 'D'), card('2', 'C'), # Initial deal: P=A,A, D=5 (hole 2)
        card('K', 'H'), # Card for first split hand (A+K=21)
        card('J', 'D'), # Card for second split hand (A+J=21)
        card('Q', 'C'), # Dealer hits (5+2+10=17)
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 7 # 4 initial + 2 split + 1 dealer hit
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=True, allow_splitting=True, count_cards=False)
    # After reset: P=[AH, AD], D=[5S, 2C]
    
    # Player splits aces
    next_obs, reward, done, info = env.step(ACTION_SPLIT)
    
    assert done == True # Game should be done because split aces auto-stand
    assert reward == 2.0 # Corrected: Expect final reward as game resolves immediately
    assert len(env.player_hands) == 2
    assert env.player_hands[0].cards == [card('A', 'H'), card('K', 'H')] # First hand (A+K=21)
    assert env.player_hands[1].cards == [card('A', 'D'), card('J', 'D')] # Second hand (A+J=21)
    assert env.player_hands[0].is_split_ace == True
    assert env.player_hands[1].is_split_ace == True
    assert env.player_hands[0].stood == True # First hand auto-stood
    assert env.player_hands[1].stood == True # Second hand auto-stood
    assert env.current_hand_index == 2 # Advanced past all player hands
    assert next_obs == (21, 5, 1) # Observation for the final state (from _get_obs fallback or last hand)

    # The game should be done, so no further steps are needed or expected.
    # The final reward should be calculated by _advance_to_next_hand_or_resolve_game
    # which is called when done becomes True.
    # Dealer has 5+2=7, hits with Q (17) -> stands.
    # Hand 1 (21) vs Dealer (17) -> Win (1.0)
    # Hand 2 (21) vs Dealer (17) -> Win (1.0)
    # The reward returned by step is the sum of all hand rewards if done is True.
    assert env.player_hands[0].reward == 1.0
    assert env.player_hands[1].reward == 1.0

# --- Test Invalid Action Handling ---
@patch('src.env.env.Deck')
def test_step_invalid_action(MockDeck):
    """Test that an invalid action penalizes the player and resolves the hand."""
    mock_deck_instance = MockDeck.return_value
    mock_deck_instance.deal_card.side_effect = [
        card('7', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal: P=15, D=5
        card('K', 'H'), card('3', 'S') # Additional cards for dealer to play out
    ]
    mock_deck_instance.cards_remaining.return_value = (1 * 52) - 6 # Adjusted for more dealer cards
    env = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)
    # After reset: P=15, D=5
    
    # Attempt an invalid action (e.g., Double Down when not allowed)
    with patch('src.env.env.logger.warning') as mock_warning: # Mock logger.warning
        next_obs, reward, done, info = env.step(ACTION_DOUBLE_DOWN) # Invalid action
        # Expect only one warning for the invalid action itself.
        # The _get_obs fallback warning is now removed from _get_obs logic for resolved states.
        mock_warning.assert_called_once() 

    assert done == True # Hand should be resolved
    assert reward == -1.0 # Penalized for invalid move
    assert env.player_hands[0].reward == -1.0
    assert env.player_hands[0].stood == True # Hand forced to stand
    assert next_obs == (15, 5, 0) # Observation remains for the hand before invalid action

# --- Test _dealer_plays helper method ---
@patch('src.env.env.Deck')
def test_dealer_plays_stands(MockDeck):
    """Test dealer stands on 17+."""
    mock_deck_instance = MockDeck.return_value
    # Provide dummy cards for initial env setup (4 cards), then no more cards for dealer hits
    mock_deck_instance.deal_card.side_effect = [
        card('2', 'H'), card('2', 'D'), card('2', 'C'), card('2', 'S') # Dummy cards for init
    ]
    env = CustomBlackjackEnv(num_decks=1, count_cards=False)
    env.dealer_hand = [card('10', 'S'), card('7', 'C')] # Force dealer hand to 17
    
    env._dealer_plays()
    dealer_sum, _ = env._update_hand_value(env.dealer_hand)
    assert dealer_sum == 17
    assert len(env.dealer_hand) == 2 # Dealer should not have hit

@patch('src.env.env.Deck')
def test_dealer_plays_hits_and_stands(MockDeck):
    """Test dealer hits then stands."""
    mock_deck_instance = MockDeck.return_value
    # Provide dummy cards for initial env setup (4 cards), then one card for dealer hit
    mock_deck_instance.deal_card.side_effect = [
        card('2', 'H'), card('2', 'D'), card('2', 'C'), card('2', 'S'), # Dummy cards for init
        card('K', 'H') # Dealer hits (5+2=7 -> 7+10=17)
    ]
    env = CustomBlackjackEnv(num_decks=1, count_cards=False)
    env.dealer_hand = [card('5', 'S'), card('2', 'C')] # Force dealer hand to 7
    
    env._dealer_plays()
    dealer_sum, _ = env._update_hand_value(env.dealer_hand)
    assert dealer_sum == 17
    assert len(env.dealer_hand) == 3 # Dealer should have hit once
    assert env.dealer_hand[2] == card('K', 'H')

@patch('src.env.env.Deck')
def test_dealer_plays_hits_and_busts(MockDeck):
    """Test dealer hits and busts."""
    mock_deck_instance = MockDeck.return_value
    # Provide dummy cards for initial env setup (4 cards), then two cards for dealer hits
    mock_deck_instance.deal_card.side_effect = [
        card('2', 'H'), card('2', 'D'), card('2', 'C'), card('2', 'S'), # Dummy cards for init
        card('5', 'H'), # Dealer hits (5+2=7 -> 7+5=12)
        card('Q', 'D')  # Dealer hits (12+10=22) - BUST
    ]
    env = CustomBlackjackEnv(num_decks=1, count_cards=False)
    env.dealer_hand = [card('5', 'S'), card('2', 'C')] # Force dealer hand to 7
    
    env._dealer_plays()
    dealer_sum, _ = env._update_hand_value(env.dealer_hand)
    assert dealer_sum > 21 # Dealer busted
    assert len(env.dealer_hand) == 4 # Dealer should have hit twice

@patch('src.env.env.Deck')
def test_dealer_plays_soft_17_hit(MockDeck):
    """Test dealer hits on soft 17 if configured."""
    mock_deck_instance = MockDeck.return_value
    # Provide dummy cards for initial env setup (4 cards), then one card for dealer hit
    mock_deck_instance.deal_card.side_effect = [
        card('2', 'H'), card('2', 'D'), card('2', 'C'), card('2', 'S'), # Dummy cards for init
        card('2', 'H') # Dealer hits (17+2=19)
    ]
    env = CustomBlackjackEnv(num_decks=1, count_cards=False, dealer_hits_on_soft_17=True)
    env.dealer_hand = [card('A', 'S'), card('6', 'C')] # Force dealer hand to soft 17
    
    env._dealer_plays()
    dealer_sum, usable_ace = env._update_hand_value(env.dealer_hand)
    assert dealer_sum == 19
    assert usable_ace # Ace is still usable as 11 if sum is 19 and not busted
    assert len(env.dealer_hand) == 3 # Dealer should have hit once

@patch('src.env.env.Deck')
def test_dealer_plays_soft_17_stands(MockDeck):
    """Test dealer stands on soft 17 if configured."""
    mock_deck_instance = MockDeck.return_value
    # Provide dummy cards for initial env setup (4 cards), then no more cards for dealer hits
    mock_deck_instance.deal_card.side_effect = [
        card('2', 'H'), card('2', 'D'), card('2', 'C'), card('2', 'S') # Dummy cards for init
    ]
    env = CustomBlackjackEnv(num_decks=1, count_cards=False, dealer_hits_on_soft_17=False)
    env.dealer_hand = [card('A', 'S'), card('6', 'C')] # Force dealer hand to soft 17
    
    env._dealer_plays()
    dealer_sum, usable_ace = env._update_hand_value(env.dealer_hand)
    assert dealer_sum == 17
    assert usable_ace # Ace is still usable as 11
    assert len(env.dealer_hand) == 2 # Dealer should not have hit

# --- Test _calculate_reward helper method ---
def test_calculate_reward_player_win_dealer_busts(basic_env):
    """Test reward when dealer busts."""
    basic_env.player_hands[0].cards = [card('10', 'H'), card('8', 'D')] # P=18
    basic_env.dealer_hand = [card('10', 'S'), card('5', 'C'), card('7', 'D')] # D=22 (bust)
    reward = basic_env._calculate_reward(basic_env.player_hands[0].cards)
    assert reward == 1.0

def test_calculate_reward_player_win_higher_sum(basic_env):
    """Test reward when player has higher sum."""
    basic_env.player_hands[0].cards = [card('10', 'H'), card('9', 'D')] # P=19
    basic_env.dealer_hand = [card('10', 'S'), card('7', 'C')] # D=17
    reward = basic_env._calculate_reward(basic_env.player_hands[0].cards)
    assert reward == 1.0

def test_calculate_reward_player_lose_dealer_higher_sum(basic_env):
    """Test reward when dealer has higher sum."""
    basic_env.player_hands[0].cards = [card('10', 'H'), card('5', 'D')] # P=15
    basic_env.dealer_hand = [card('10', 'S'), card('8', 'C')] # D=18
    reward = basic_env._calculate_reward(basic_env.player_hands[0].cards)
    assert reward == -1.0

def test_calculate_reward_push(basic_env):
    """Test reward for a push."""
    basic_env.player_hands[0].cards = [card('10', 'H'), card('7', 'D')] # P=17
    basic_env.dealer_hand = [card('10', 'S'), card('7', 'C')] # D=17
    reward = basic_env._calculate_reward(basic_env.player_hands[0].cards)
    assert reward == 0.0

# --- Test Seeding for Reproducibility ---
def test_env_reset_reproducibility():
    """
    Test that resetting the environment with the same seed produces the same initial state
    (cards dealt, running count if applicable).
    """
    # Define a factory function for mock decks
    def create_mock_deck_with_sequence(cards_sequence):
        mock_deck_instance = MagicMock(spec=Deck)
        mock_deck_instance.deal_card.side_effect = list(cards_sequence) # Use a copy of the list
        mock_deck_instance.cards_remaining.return_value = (1 * 52) - len(cards_sequence) # Adjust as needed
        mock_deck_instance.running_count = 0 # Ensure mock has this attribute
        return mock_deck_instance

    # Sequence for initial reset for both envs
    initial_deal_sequence = [
        card('2', 'H'), card('7', 'S'), card('3', 'D'), card('K', 'C')
    ]

    # Patch Deck to return a new mock instance with the defined sequence each time it's called
    with patch('src.env.env.Deck', side_effect=lambda num_decks, seed, reshuffle_threshold_pct: create_mock_deck_with_sequence(initial_deal_sequence)) as MockDeckClass:
        env1 = CustomBlackjackEnv(num_decks=1, count_cards=True)
        env2 = CustomBlackjackEnv(num_decks=1, count_cards=True)

        # Resetting explicitly, which will cause CustomBlackjackEnv to create a NEW Deck
        # This new Deck will also be a mock with the same initial_deal_sequence
        obs1, info1 = env1.reset(23)
        obs2, info2 = env2.reset(23)

        assert obs1 == obs2
        assert info1 == info2
        assert env1.player_hands[0].cards == env2.player_hands[0].cards
        assert env1.dealer_hand == env2.dealer_hand
        assert env1.running_count == env2.running_count

def test_env_step_reproducibility():
    """
    Test that stepping through the environment with the same seed and actions
    produces the same sequence of observations, rewards, and states.
    """
    # Define the full sequence of cards needed for each environment's lifecycle
    # 4 cards for initial deal, 1 for player hit, 1 for dealer hit
    full_sequence_for_one_env = [
        card('7', 'H'), card('5', 'S'), card('8', 'D'), card('2', 'C'), # Initial deal
        card('A', 'H'), # Player hits (15+11=26) - BUST
        card('K', 'S'), # Dealer hits (5+2=7 -> 7+10=17)
    ]

    # Define a factory function for mock decks for this specific test
    def create_mock_deck_for_step_test(num_decks, seed, reshuffle_threshold_pct):
        mock_deck_instance = MagicMock(spec=Deck)
        mock_deck_instance.deal_card.side_effect = list(full_sequence_for_one_env) # Each mock gets its own copy
        mock_deck_instance.cards_remaining.return_value = (num_decks * 52) - len(full_sequence_for_one_env)
        mock_deck_instance.running_count = 0 # Ensure mock has this attribute
        return mock_deck_instance

    # Patch Deck to return a new mock instance with the defined sequence each time it's called
    with patch('src.env.env.Deck', side_effect=create_mock_deck_for_step_test) as MockDeckClass:
        env1 = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)
        env2 = CustomBlackjackEnv(num_decks=1, allow_doubling=False, allow_splitting=False, count_cards=False)

        # Perform the same action sequence
        action = ACTION_HIT # Player hits
        obs1, reward1, done1, info1 = env1.step(action)
        obs2, reward2, done2, info2 = env2.step(action)

        assert obs1 == obs2
        assert reward1 == reward2
        assert done1 == done2
        assert info1 == info2
        assert env1.player_hands[0].cards == env2.player_hands[0].cards
        assert env1.dealer_hand == env2.dealer_hand
        assert env1.running_count == env2.running_count # Should be 0 if count_cards=False

        # If not done, perform another step
        if not done1:
            action = ACTION_STAND # Player stands
            obs1, reward1, done1, info1 = env1.step(action)
            obs2, reward2, done2, info2 = env2.step(action)

            assert obs1 == obs2
            assert reward1 == reward2
            assert done1 == done2
            assert info1 == info2
            assert env1.player_hands[0].cards == env2.player_hands[0].cards
            assert env1.dealer_hand == env2.dealer_hand
            assert env1.running_count == env2.running_count
