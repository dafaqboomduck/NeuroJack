import numpy as np
import random
import logging
from typing import List, Tuple, Dict, Union, Any, Optional

# Assuming Card, Deck, PlayerHand are imported from their respective files
from src.env.playerhand import PlayerHand
from src.env.deck import Deck
from src.env.card import Card

# Configure logging for the environment
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # Set to INFO for general messages, DEBUG for detailed tracing

# Define action constants for clarity
ACTION_STAND = 0
ACTION_HIT = 1
ACTION_DOUBLE_DOWN = 2
ACTION_SPLIT = 3

class CustomBlackjackEnv:
    """
    Custom Blackjack Environment with additional rules and card counting,
    without relying on Gymnasium.

    Observation Space:
    Tuple: (player_current_sum, dealer_card_showing, usable_ace[, running_count, true_count])
    - player_current_sum: Sum of player's current hand (int, 2–22).\n
    - dealer_card_showing: Value of dealer's visible card (int, 1–11, Ace=11).\n
    - usable_ace: Whether player has a usable ace (int, 0 or 1).\n
    - running_count: Current Hi-Lo running count (if count_cards=True)\n
    - true_count: Current Hi-Lo true count (running_count / decks_remaining) (if count_cards=True)

    Action Space:
    0: Stand
    1: Hit
    2: Double Down (if allowed)
    3: Split (if allowed)
    """

    def __init__(self, render_mode: Optional[str] = None, num_decks: int = 6, blackjack_payout: float = 1.5,
                 allow_doubling: bool = True, allow_splitting: bool = True, count_cards: bool = True,
                 dealer_hits_on_soft_17: bool = True,
                 reshuffle_threshold_pct: float = 0.25):
        self.num_decks = num_decks
        self.blackjack_payout = blackjack_payout
        self.allow_doubling = allow_doubling
        self.allow_splitting = allow_splitting
        self.count_cards = count_cards
        # Removed self.seed as it's no longer used for reproducibility
        self.render_mode = render_mode
        self.dealer_hits_on_soft_17 = dealer_hits_on_soft_17
        self.reshuffle_threshold_pct = reshuffle_threshold_pct

        self.observation_description = (
            "(player_current_sum, dealer_card_showing, usable_ace"
            + (", running_count, true_count)" if self.count_cards else ")")
        )
        actions = ["0: Stand", "1: Hit"]
        if self.allow_doubling:
            actions.append("2: Double Down")
        if self.allow_splitting:
            actions.append("3: Split")
        self.action_description = ", ".join(actions)

        # Initialize Deck without a seed, so it uses system randomness
        self.deck: Deck = Deck(self.num_decks, seed=None, reshuffle_threshold_pct=self.reshuffle_threshold_pct)
        self.player_hands: List[PlayerHand] = []
        self.dealer_hand: List[Card] = []
        self.current_hand_index: int = 0
        self.running_count: int = 0

        self.reset()

    @property
    def state_size(self) -> int:
        # This property dynamically returns the correct state size
        return 5 if self.count_cards else 3

    @property
    def num_actions(self) -> int:
        # Always 0 and 1 (Stand, Hit), and conditionally Double Down and Split
        return 2 + int(self.allow_doubling) + int(self.allow_splitting)

    def _update_hand_value(self, hand_cards: List[Card]) -> Tuple[int, bool]:
        hand_sum_soft = 0 # Sum with all Aces as 11 initially
        num_aces_in_hand = 0
        for card in hand_cards:
            if card.rank == 'A':
                num_aces_in_hand += 1
                hand_sum_soft += 11
            else:
                hand_sum_soft += card.value

        # Adjust aces from 11 to 1 if busting
        current_sum = hand_sum_soft
        aces_remaining_as_11 = num_aces_in_hand

        while current_sum > 21 and aces_remaining_as_11 > 0:
            current_sum -= 10 # Convert an Ace from 11 to 1
            aces_remaining_as_11 -= 1 # One less ace contributing 11

        # A usable ace exists if there was at least one ace originally,
        # and after adjustments, at least one ace is still counted as 11.
        # This is equivalent to: current_sum <= 21 AND aces_remaining_as_11 > 0
        usable_ace = (current_sum <= 21 and aces_remaining_as_11 > 0)

        return current_sum, usable_ace

    def _deal_card(self, hand_obj_or_list: Union[PlayerHand, List[Card]], face_up: bool = True, is_initial_deal: bool = False) -> Card:
        card = self.deck.deal_card()
        if isinstance(hand_obj_or_list, PlayerHand):
            hand_obj_or_list.add_card(card)
        else: # Dealer's hand is a list of Card objects
            hand_obj_or_list.append(card)

        # Update running count only for face-up cards, unless it's the initial deal reset logic
        if self.count_cards and face_up and not is_initial_deal:
            self.running_count += card.count_value
            logger.debug(f"Dealt {card}, running count updated to {self.running_count}")
        elif self.count_cards and not face_up and not is_initial_deal:
            # For dealer's hole card, count it when it's revealed at dealer's turn
            logger.debug(f"Dealt {card} face down.")

        return card

    def _get_obs(self) -> Union[Tuple[int, int, int], Tuple[int, int, int, int, int]]:
        # This method should always be called with a valid current_hand_index
        # or when the game is fully resolved and we need a final observation.
        if not (0 <= self.current_hand_index < len(self.player_hands)):
            # If all player hands are resolved, this is a valid state for _get_obs to be called
            # to get the final observation. No warning is needed here.
            player_sum_for_obs = self._update_hand_value(self.player_hands[0].cards)[0] if self.player_hands else 0
            dealer_showing_value = self.dealer_hand[0].value
            usable_ace_for_obs = self._update_hand_value(self.player_hands[0].cards)[1] if self.player_hands else 0

            obs: Tuple[int, ...] = (player_sum_for_obs, dealer_showing_value, int(usable_ace_for_obs))
            if self.count_cards:
                decks_remaining = max(1e-6, self.deck.cards_remaining() / 52.0)
                true_count = round(self.running_count / decks_remaining)
                obs += (self.running_count, true_count)
            return obs


        current_player_hand_cards = self.player_hands[self.current_hand_index].cards
        player_sum, usable_ace = self._update_hand_value(current_player_hand_cards)
        dealer_showing_value = self.dealer_hand[0].value

        obs: Tuple[int, ...] = (player_sum, dealer_showing_value, int(usable_ace))
        if self.count_cards:
            decks_remaining = max(1e-6, self.deck.cards_remaining() / 52.0)
            true_count = round(self.running_count / decks_remaining)
            obs += (self.running_count, true_count)

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Tuple[int, ...], Dict[str, bool]]:
        # Removed seed parameter and logic from reset, as the environment will now be truly random
        # Re-initialize Deck without a seed for each reset
        self.deck = Deck(self.num_decks, seed=None, reshuffle_threshold_pct=self.reshuffle_threshold_pct)

        self.player_hands = [PlayerHand()]
        self.dealer_hand = []
        self.current_hand_index = 0
        self.running_count = 0 # Reset running count on new shoe

        # Initial deal - deal cards but don't update running_count yet,
        # calculate it explicitly for initial face-up cards
        self._deal_card(self.player_hands[0], face_up=True, is_initial_deal=True) # Player Card 1
        self._deal_card(self.dealer_hand, face_up=True, is_initial_deal=True)      # Dealer Up Card
        self._deal_card(self.player_hands[0], face_up=True, is_initial_deal=True) # Player Card 2
        self._deal_card(self.dealer_hand, face_up=False, is_initial_deal=True)     # Dealer Hole Card

        if self.count_cards:
            # Explicitly calculate running count for initial face-up cards
            self.running_count += self.player_hands[0].cards[0].count_value
            self.running_count += self.player_hands[0].cards[1].count_value
            self.running_count += self.dealer_hand[0].count_value
            logger.debug(f"Reset: Initial running count: {self.running_count}")

        player_sum, _ = self._update_hand_value(self.player_hands[0].cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand) # Includes hole card for blackjack check

        done = False
        reward = 0.0
        info: Dict[str, bool] = {"can_double": False, "can_split": False}

        player_blackjack = (player_sum == 21 and len(self.player_hands[0].cards) == 2)
        dealer_blackjack = (dealer_sum == 21 and len(self.dealer_hand) == 2)

        if player_blackjack and dealer_blackjack:
            reward = 0.0
            done = True
            logger.info("Reset: Push (both blackjack).")
        elif player_blackjack:
            reward = self.blackjack_payout
            done = True
            logger.info(f"Reset: Player Blackjack! Reward: {reward}")
        elif dealer_blackjack:
            reward = -1.0
            done = True
            # Reveal dealer hole card if they have blackjack at start
            if self.count_cards:
                self.running_count += self.dealer_hand[1].count_value
                logger.debug(f"Dealer Blackjack: Hole card revealed, running count: {self.running_count}")
            logger.info(f"Reset: Dealer Blackjack. Reward: {reward}")

        if not done:
            if self.allow_doubling:
                info["can_double"] = True
            if self.allow_splitting and self.player_hands[0].cards[0].rank == self.player_hands[0].cards[1].rank:
                info["can_split"] = True

        if done:
            self.player_hands[0].reward = reward
            # FIX: Set player hand to stood if game ends on initial deal (blackjack, push, or dealer blackjack)
            self.player_hands[0].stood = True 

        observation = self._get_obs() # Get the initial observation based on the current state
        if self.render_mode == 'human':
            self.render()
        logger.info(f"Environment reset. Observation: {observation}, Info: {info}, Done: {done}")
        return observation, info

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, bool]]:
        reward = 0.0
        done = False
        info: Dict[str, bool] = {"can_double": False, "can_split": False}

        if not (0 <= self.current_hand_index < len(self.player_hands)):
            logger.error("Step called when no active player hands remain. This should not happen.")
            # If this state is reached, the game should already be done, so return final obs.
            return self._get_obs(), 0.0, True, info

        current_player_hand_obj = self.player_hands[self.current_hand_index]
        current_player_hand_cards = current_player_hand_obj.cards
        
        is_first_action = (len(current_player_hand_cards) == 2 and
                           not current_player_hand_obj.stood and
                           not current_player_hand_obj.double_down)

        current_hand_resolved = False # Flag to indicate if current hand's play is finished

        if action == ACTION_HIT:
            logger.debug(f"Hand {self.current_hand_index + 1}: Player hits.")
            self._deal_card(current_player_hand_obj, face_up=True)
            player_sum, _ = self._update_hand_value(current_player_hand_obj.cards)
            if player_sum > 21:
                current_player_hand_obj.reward = -1.0
                if current_player_hand_obj.double_down:
                    current_player_hand_obj.reward *= 2
                logger.info(f"Hand {self.current_hand_index + 1}: Player busts ({player_sum}). Reward: {current_player_hand_obj.reward}")
                current_player_hand_obj.stood = True # Mark hand as stood when it busts
                current_hand_resolved = True
            # Else, hand is not resolved yet, player can continue to hit/stand
        elif action == ACTION_STAND:
            logger.debug(f"Hand {self.current_hand_index + 1}: Player stands ({self._update_hand_value(current_player_hand_obj.cards)[0]}).")
            current_player_hand_obj.stood = True
            current_hand_resolved = True
        elif action == ACTION_DOUBLE_DOWN and self.allow_doubling and is_first_action:
            logger.debug(f"Hand {self.current_hand_index + 1}: Player doubles down.")
            self._deal_card(current_player_hand_obj, face_up=True)
            player_sum, _ = self._update_hand_value(current_player_hand_obj.cards)
            current_player_hand_obj.double_down = True
            current_player_hand_obj.stood = True # Automatically stands after double down
            if player_sum > 21:
                current_player_hand_obj.reward = -1.0 * 2 # Double penalty for bust on double down
                logger.info(f"Hand {self.current_hand_index + 1}: Player busts on double down ({player_sum}). Reward: {current_player_hand_obj.reward}")
            current_hand_resolved = True
        elif action == ACTION_SPLIT and self.allow_splitting and is_first_action and \
             current_player_hand_cards[0].rank == current_player_hand_cards[1].rank:
            logger.debug(f"Hand {self.current_hand_index + 1}: Player splits.")
            card1, card2 = current_player_hand_cards

            # Clear current hand and add one card back
            current_player_hand_obj.cards = [card1]
            # Create new hand for the second card
            new_hand = PlayerHand(cards=[card2])

            # Special rule for splitting Aces
            if card1.rank == 'A':
                current_player_hand_obj.is_split_ace = True
                new_hand.is_split_ace = True
                logger.debug("Splitting Aces detected.")

            # Insert new hand right after the current one for sequential play
            self.player_hands.insert(self.current_hand_index + 1, new_hand)

            # Deal one card to each new hand
            self._deal_card(current_player_hand_obj, face_up=True)
            self._deal_card(new_hand, face_up=True)

            # If Aces were split, automatically stand these hands
            if current_player_hand_obj.is_split_ace:
                current_player_hand_obj.stood = True
                if new_hand.is_split_ace:
                    new_hand.stood = True # Ensure the newly created split ace hand is also stood

            # After split, the current hand (first split hand) is still active for actions
            # unless it was an Ace split.
            if not current_player_hand_obj.stood: # Only if not auto-stood (i.e., not split aces)
                current_hand_resolved = False # Player can still hit/stand on this hand
            else: # If it was an Ace split, it's auto-stood
                current_hand_resolved = True
        else:
            # Invalid action
            logger.warning(f"Hand {self.current_hand_index + 1}: Invalid action {action} performed. Penalizing.")
            current_player_hand_obj.reward = -1 # Penalty for illegal move
            current_player_hand_obj.stood = True # Forcing hand to stand after invalid move
            current_hand_resolved = True

        # Now, advance to the next hand if the current one is resolved, or resolve the game.
        if current_hand_resolved:
            done = self._advance_to_next_hand_or_resolve_game()
        else:
            done = False # Game is not done yet if hand is not resolved

        # Determine the observation *after* all state changes, including advancing to the next hand.
        next_observation = self._get_obs()

        # Update info for next observation (if game is not done)
        if not done and self.current_hand_index < len(self.player_hands):
            next_hand = self.player_hands[self.current_hand_index]
            next_hand_cards = next_hand.cards
            # Recalculate player_next_sum and usable_ace for the *next* active hand
            # (already done by _get_obs, but info requires this logic)
            # This logic is redundant with _get_obs, but kept for clarity of info dict.
            player_next_sum, _ = self._update_hand_value(next_hand_cards)

            info["can_double"] = (self.allow_doubling and
                                  len(next_hand_cards) == 2 and
                                  not next_hand.stood and
                                  not next_hand.double_down and
                                  not next_hand.is_split_ace)

            info["can_split"] = (self.allow_splitting and
                                 len(next_hand_cards) == 2 and
                                 next_hand_cards[0].rank == next_hand_cards[1].rank and
                                 not next_hand.stood and
                                 not next_hand.double_down)
        else:
            # If done, no actions are possible
            info = {"can_double": False, "can_split": False}


        final_reward = sum(hand.reward for hand in self.player_hands) if done else 0.0

        if self.render_mode == 'human':
            self.render()

        logger.debug(f"Step completed. Action: {action}, Next Obs: {next_observation}, Final Reward: {final_reward}, Done: {done}, Info: {info}")
        return next_observation, final_reward, done, info

    def _advance_to_next_hand_or_resolve_game(self) -> bool:
        """
        Advances the current_hand_index to the next active player hand.
        If all player hands are resolved, the dealer plays and game ends.
        Returns True if the game is done, False otherwise.
        """
        while self.current_hand_index < len(self.player_hands) and self.player_hands[self.current_hand_index].stood:
            self.current_hand_index += 1

        if self.current_hand_index >= len(self.player_hands):
            # All player hands are done, dealer plays
            logger.info("All player hands resolved. Dealer's turn.")
            self._dealer_plays()
            # Calculate final rewards for all hands that didn't bust
            for i, hand in enumerate(self.player_hands):
                if hand.reward == 0.0: # Only calculate if not already busted or penalized
                    hand.reward = self._calculate_reward(hand.cards)
                    if hand.double_down:
                        hand.reward *= 2
                    logger.debug(f"Hand {i+1} final reward: {hand.reward}")
            return True # Game is done
        return False # Not all player hands are done yet

    def _dealer_plays(self) -> None:
        """Dealer hits until sum is 17 or more."""
        # Reveal dealer's hole card and update count
        if self.count_cards and len(self.dealer_hand) == 2:
            self.running_count += self.dealer_hand[1].count_value
            logger.debug(f"Dealer hole card revealed: {self.dealer_hand[1]}, running count: {self.running_count}")

        dealer_sum, usable_ace = self._update_hand_value(self.dealer_hand)
        logger.info(f"Dealer starts playing with {dealer_sum} (usable ace: {usable_ace}).")

        while True:
            if dealer_sum > 21:
                logger.info(f"Dealer busts with {dealer_sum}.")
                break
            # Dealer hits on 16 or less
            # Dealer hits on soft 17 if dealer_hits_on_soft_17 is True
            if dealer_sum < 17 or (dealer_sum == 17 and usable_ace and self.dealer_hits_on_soft_17):
                logger.debug(f"Dealer hits (current sum: {dealer_sum}, usable ace: {usable_ace}).")
                self._deal_card(self.dealer_hand, face_up=True)
                dealer_sum, usable_ace = self._update_hand_value(self.dealer_hand)
            else:
                logger.info(f"Dealer stands with {dealer_sum}.")
                break

    def _calculate_reward(self, player_hand_cards: List[Card]) -> float:
        player_sum, _ = self._update_hand_value(player_hand_cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand)

        logger.info(f"Calculating reward for player hand with sum {player_sum} against dealer sum {dealer_sum}. Dealer total: {dealer_sum}") # Added log

        # Player bust is handled immediately in step, so this is for hands that didn't bust.
        if player_sum > 21: # Should ideally not happen if called correctly
            logger.error("Error: _calculate_reward called for a busted hand that should have been resolved earlier.") # Added log
            return -1.0
        elif dealer_sum > 21:
            logger.info("Dealer busted. Player wins.") # Added log
            return 1.0 # Player wins because dealer busted
        elif player_sum > dealer_sum:
            logger.info("Player has higher sum. Player wins.") # Added log
            return 1.0
        elif player_sum < dealer_sum:
            logger.info("Dealer has higher sum. Player loses.") # Added log
            return -1.0
        else:
            logger.info("Player and Dealer have same sum. Push.") # Added log
            return 0.0 # Push

    def render(self) -> None:
        if self.render_mode != 'human':
            return

        print("\n--- Blackjack Game ---")
        for i, hand in enumerate(self.player_hands):
            hand_sum, usable_ace = self._update_hand_value(hand.cards)
            status = []
            if hand.stood: status.append("Stood")
            if hand.double_down: status.append("Doubled Down")
            if hand.is_split_ace: status.append("Split Ace Hand")
            if hand_sum > 21: status.append("Bust")
            if hand.reward != 0: status.append(f"Reward: {hand.reward:.2f}")

            status_str = ", ".join(status) if status else "Active"
            print(f"Player Hand {i+1} ({'Current' if i == self.current_hand_index else 'Other'}): "
                  f"{[str(c) for c in hand.cards]} (Sum: {hand_sum}, Usable Ace: {int(usable_ace)}) [{status_str}]")

        dealer_sum, _ = self._update_hand_value(self.dealer_hand)
        # Only show hole card if game is over or dealer is playing
        if len(self.dealer_hand) == 2 and self.current_hand_index < len(self.player_hands):
            dealer_cards_display = [str(self.dealer_hand[0]), '??']
            dealer_total_display = '??'
        else:
            dealer_cards_display = [str(c) for c in self.dealer_hand]
            dealer_total_display = dealer_sum
        print(f"Dealer Hand: {dealer_cards_display} (Showing: {self.dealer_hand[0].value}, Total: {dealer_total_display})")

        if self.count_cards:
            decks_remaining = max(1e-6, self.deck.cards_remaining() / 52.0)
            true_count = round(self.running_count / decks_remaining)
            print(f"Running Count: {self.running_count}, True Count: {true_count} (Decks Left: {decks_remaining:.1f})")
        print("----------------------")

    def close(self) -> None:
        pass
