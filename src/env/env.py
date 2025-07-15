import numpy as np
from src.env.playerhand import PlayerHand
from src.env.deck import Deck
import random

class CustomBlackjackEnv:
    """
    Custom Blackjack Environment with additional rules and card counting,
    without relying on Gymnasium.

    Observation Space:
    A tuple: (player_current_sum, dealer_card_showing, usable_ace, running_count)
    - player_current_sum: Sum of player's current hand (int, 2-22).
    - dealer_card_showing: Value of dealer's visible card (int, 1-11, Ace=11).
    - usable_ace: Whether player has a usable ace (int, 0 or 1).
    - running_count: Current Hi-Lo running count (int, range depends on num_decks).

    Action Space:
    An integer:
    0: Stand
    1: Hit
    2: Double Down (only on first move, if applicable)
    3: Split (only on first move with two same-rank cards, if applicable)
    """

    def __init__(self, render_mode=None, num_decks=6, blackjack_payout=1.5,
                 allow_doubling=True, allow_splitting=True, count_cards=True, seed=None):
        self.num_decks = num_decks
        self.blackjack_payout = blackjack_payout
        self.allow_doubling = allow_doubling
        self.allow_splitting = allow_splitting
        self.count_cards = count_cards
        self.seed = seed # Store seed for reproducibility

        # Define descriptive attributes for observation and action spaces
        self.observation_description = "(player_current_sum, dealer_card_showing, usable_ace, running_count)"
        self.action_description = "0: Stand, 1: Hit, 2: Double Down, 3: Split"

        self.deck = Deck(self.num_decks, seed=self.seed) # Pass seed to deck
        self.player_hands = [] # List of PlayerHand objects (for handling splits)
        self.dealer_hand = []  # List of Card objects
        self.current_hand_index = 0 # Index of the player's hand currently being played
        self.running_count = 0 # Current Hi-Lo running count
        self.render_mode = render_mode

        self.reset() # Initialize the environment state

    def _update_hand_value(self, hand_cards):
        """
        Calculates the numerical value of a hand and determines if it has a usable ace.
        An ace is 'usable' if it can be counted as 11 without the hand busting.
        """
        hand_sum = 0
        num_aces = 0
        for card in hand_cards:
            if card.rank == 'A':
                num_aces += 1
            hand_sum += card.value # Initially count Ace as 11

        usable_ace = False
        # An ace is usable if it can be 11 without busting
        if num_aces > 0 and hand_sum <= 21:
            usable_ace = True

        # If hand busts with Aces as 11, convert them to 1 until it's no longer busted or no more Aces
        while hand_sum > 21 and num_aces > 0:
            hand_sum -= 10 # Convert an Ace from 11 to 1
            num_aces -= 1
            usable_ace = False # Once an ace is converted to 1, it's no longer 'usable' as 11

        return hand_sum, usable_ace

    def _deal_card(self, hand_obj_or_list, is_player=True, face_up=True):
        """
        Deals a card from the deck to a hand and updates the running count
        if the card is face-up.
        """
        card = self.deck.deal_card()
        if isinstance(hand_obj_or_list, PlayerHand):
            hand_obj_or_list.add_card(card)
        else: # Must be dealer_hand (list of Cards)
            hand_obj_or_list.append(card)

        if self.count_cards and face_up: # Only count cards that are visible to the player
            self.running_count += card.count_value
        return card

    def _get_obs(self):
        """
        Returns the current observation tuple for the active player hand.
        This is the state an RL agent would observe.
        """
        # This check is crucial to prevent IndexError in terminal states
        if not (0 <= self.current_hand_index < len(self.player_hands)):
            # This case should ideally be handled by the caller (step method)
            # but as a safeguard, return a 'zero' observation if no valid hand
            return (0, 0, 0, 0)

        current_player_hand_cards = self.player_hands[self.current_hand_index].cards
        player_sum, usable_ace = self._update_hand_value(current_player_hand_cards)
        
        dealer_showing_card = self.dealer_hand[0] # Dealer's first card is always face up
        dealer_showing_value = dealer_showing_card.value # Ace is 11 here, consistent with Gym's Blackjack-v1
        
        return (player_sum, dealer_showing_value, int(usable_ace), self.running_count)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state for a new episode.
        """
        if seed is not None:
            self.seed = seed
            random.seed(self.seed) # Set seed for random module
            self.deck = Deck(self.num_decks, seed=self.seed) # Recreate deck with new seed
        else:
            self.deck = Deck(self.num_decks, seed=self.seed) # Recreate and shuffle deck for a fresh start

        self.player_hands = [PlayerHand()] # Start with one player hand
        self.dealer_hand = []
        self.current_hand_index = 0 # Start with the first hand
        self.running_count = 0 # Reset card count

        # Deal initial cards: two to player, two to dealer (one face up, one face down)
        self._deal_card(self.player_hands[0], is_player=True, face_up=True)
        self._deal_card(self.dealer_hand, is_player=False, face_up=True) # Dealer's up card
        self._deal_card(self.player_hands[0], is_player=True, face_up=True)
        self._deal_card(self.dealer_hand, is_player=False, face_up=False) # Dealer's hole card (not counted initially)

        # Update running count for initial visible cards (player's two, dealer's up card)
        if self.count_cards:
            self.running_count = 0 # Reset before counting initial visible cards
            self.running_count += self.player_hands[0].cards[0].count_value
            self.running_count += self.player_hands[0].cards[1].count_value
            self.running_count += self.dealer_hand[0].count_value # Dealer's up card

        # Check for immediate blackjacks after initial deal
        player_sum, _ = self._update_hand_value(self.player_hands[0].cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand) # This will calculate with hole card

        done = False
        reward = 0
        info = {"can_double": False, "can_split": False} # Info about possible actions for the agent

        player_blackjack = (player_sum == 21 and len(self.player_hands[0].cards) == 2)
        dealer_blackjack = (dealer_sum == 21 and len(self.dealer_hand) == 2)

        # Handle immediate blackjacks
        if player_blackjack and dealer_blackjack:
            reward = 0 # Push
            done = True
        elif player_blackjack:
            reward = self.blackjack_payout # Player wins with blackjack
            done = True
        elif dealer_blackjack:
            reward = -1 # Player loses to dealer blackjack
            done = True

        # Determine if doubling/splitting is allowed for the first move of the hand
        if not done:
            if self.allow_doubling:
                info["can_double"] = True
            # Check if initial cards have same rank for splitting
            if self.allow_splitting and self.player_hands[0].cards[0].rank == self.player_hands[0].cards[1].rank:
                info["can_split"] = True
        
        # If the game is done immediately (due to blackjack), store the reward for the hand
        if done:
            self.player_hands[0].reward = reward
            
        observation = self._get_obs()
        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Takes an action in the environment.

        Args:
            action (int): The action to take (0: Stand, 1: Hit, 2: Double Down, 3: Split).

        Returns:
            tuple: (observation, reward, done, info)
                - observation (tuple): The agent's observation of the environment.
                - reward (float): The total reward received at the end of the episode.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information about the environment state.
        """
        reward = 0 # Reward for the current step (will be 0 until episode ends)
        done = False
        info = {"can_double": False, "can_split": False} # Reset info for current step

        current_player_hand_obj = self.player_hands[self.current_hand_index]
        current_player_hand_cards = current_player_hand_obj.cards
        player_sum, usable_ace = self._update_hand_value(current_player_hand_cards)

        # Flag to track if the current hand is resolved by this action
        current_hand_resolved = False

        is_first_action_on_hand = (len(current_player_hand_cards) == 2 and 
                                   not current_player_hand_obj.stood and 
                                   not current_player_hand_obj.double_down)

        if action == 1: # Hit
            self._deal_card(current_player_hand_obj, is_player=True, face_up=True)
            player_sum, usable_ace = self._update_hand_value(current_player_hand_obj.cards)
            if player_sum > 21:
                # Player busts on this hand
                current_player_hand_obj.reward = -1 # Store individual hand reward
                if current_player_hand_obj.double_down: current_player_hand_obj.reward *= 2
                current_hand_resolved = True
            else:
                # Game continues for this hand, but cannot double/split anymore after hitting
                current_player_hand_obj.stood = True # Mark as stood to prevent further double/split on this hand
        elif action == 0: # Stand
            current_player_hand_obj.stood = True
            current_hand_resolved = True
        elif action == 2: # Double Down
            if self.allow_doubling and is_first_action_on_hand:
                self._deal_card(current_player_hand_obj, is_player=True, face_up=True)
                player_sum, usable_ace = self._update_hand_value(current_player_hand_obj.cards)
                current_player_hand_obj.double_down = True
                current_player_hand_obj.stood = True # Player stands after doubling
                if player_sum > 21:
                    current_player_hand_obj.reward = -1 # Double down and bust
                    if current_player_hand_obj.double_down: current_player_hand_obj.reward *= 2
                current_hand_resolved = True
            else:
                # Invalid action: penalize and treat as if the player stood
                current_player_hand_obj.reward = -0.75 # Penalty for invalid action
                current_player_hand_obj.stood = True
                current_hand_resolved = True
        elif action == 3: # Split
            if self.allow_splitting and is_first_action_on_hand and current_player_hand_cards[0].rank == current_player_hand_cards[1].rank:
                # Create two new hands from the current hand
                card1 = current_player_hand_cards[0]
                card2 = current_player_hand_cards[1]
                
                # Modify the current hand to contain only the first card
                current_player_hand_obj.cards = [card1]
                # Insert a new hand for the second card right after the current one in the list
                new_hand = PlayerHand(cards=[card2])
                self.player_hands.insert(self.current_hand_index + 1, new_hand)

                # Deal a second card to each of the new hands
                self._deal_card(current_player_hand_obj, is_player=True, face_up=True)
                self._deal_card(new_hand, is_player=True, face_up=True)
                
                # No immediate reward, the game continues for the current (first split) hand.
                # The newly dealt cards make these hands eligible for double/split on their first action.
                current_hand_resolved = False # Explicitly mark as not resolved for split
            else:
                # Invalid action: penalize and treat as if the player stood
                current_player_hand_obj.reward = -0.75 # Penalty for invalid action
                current_player_hand_obj.stood = True
                current_hand_resolved = True
        else:
            # Handle unexpected actions - treat as stand and penalize
            current_player_hand_obj.reward = -0.75
            current_player_hand_obj.stood = True
            current_hand_resolved = True

        # Decide if the *overall episode* is done or if we advance to next hand
        if current_hand_resolved:
            done = self._advance_to_next_hand_or_resolve_game()
        else:
            # If a valid split happened, the current hand (first split hand) is still active.
            # If hit and not bust, the current hand is also still active.
            done = False # Episode is not done yet

        # Determine the `next_observation` to return
        next_observation = None
        if not done:
            # If the episode is NOT done, it means there's still an active player hand
            # (either the same hand after a non-resolving action like hit/split,
            # or the next hand after a resolved hand if more split hands exist).
            next_observation = self._get_obs()
        else:
            # If the episode IS done, it means all player hands are resolved
            # and the dealer has played. There's no "next active player hand" for _get_obs().
            # We return a 'terminal' observation, which can be zeros.
            # The agent should learn that a 'done' signal means this state is final.
            next_observation = (0, 0, 0, 0) # Placeholder for a terminal observation

        if self.render_mode == 'human':
            self.render()

        # Update info about possible actions for the *next* state, if game is continuing
        if not done:
            # Re-get current hand cards and object, as they might have changed due to split or hit
            # This is important if next_observation was taken before this info update.
            current_player_hand_cards = self.player_hands[self.current_hand_index].cards 
            current_player_hand_obj = self.player_hands[self.current_hand_index]

            # Check if doubling is possible for the current hand
            if self.allow_doubling and len(current_player_hand_cards) == 2 and not current_player_hand_obj.stood:
                info["can_double"] = True
            # Check if splitting is possible for the current hand
            if self.allow_splitting and len(current_player_hand_cards) == 2 and \
               current_player_hand_cards[0].rank == current_player_hand_cards[1].rank and not current_player_hand_obj.stood:
                info["can_split"] = True


        # The final episode reward is only calculated and returned when the entire episode is done.
        final_reward = 0
        if done:
            final_reward = sum(hand.reward for hand in self.player_hands)

        return next_observation, final_reward, done, info

    def _advance_to_next_hand_or_resolve_game(self):
        """
        Helper function to manage the flow of play after a player action.
        If there are more split hands to play, it advances to the next hand.
        If all player hands are resolved, the dealer plays and the game concludes.
        Returns True if the entire episode is done, False otherwise.
        """
        self.current_hand_index += 1
        
        if self.current_hand_index < len(self.player_hands):
            # There are more hands to play (due to splitting), so the episode is not yet done.
            return False 
        else:
            # All player hands have been played, now the dealer plays.
            self._dealer_plays()
            # Calculate rewards for all player hands that haven't already busted or been penalized.
            for hand_obj in self.player_hands:
                if hand_obj.reward == 0: # Only calculate if reward hasn't been set (e.g., by bust or invalid action)
                    hand_obj.reward = self._calculate_reward(hand_obj.cards)
                    if hand_obj.double_down:
                        hand_obj.reward *= 2 # Double the reward if the hand was doubled down
            return True # The entire episode is now done.

    def _dealer_plays(self):
        """
        Logic for the dealer's turn: hits until their hand sum is 17 or more.
        Updates the running count for the dealer's hole card when it's revealed.
        """
        # Reveal dealer's hole card and update running count
        if self.count_cards:
            # Ensure dealer_hand has at least two cards before accessing index 1
            if len(self.dealer_hand) > 1:
                self.running_count += self.dealer_hand[1].count_value # Add hole card to count

        dealer_sum, _ = self._update_hand_value(self.dealer_hand)
        while dealer_sum < 17:
            self._deal_card(self.dealer_hand, is_player=False, face_up=True)
            dealer_sum, _ = self._update_hand_value(self.dealer_hand)

    def _calculate_reward(self, player_hand_cards):
        """
        Calculates the reward for a single player hand against the dealer's hand.
        """
        player_sum, _ = self._update_hand_value(player_hand_cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand)

        player_blackjack = (player_sum == 21 and len(player_hand_cards) == 2)
        dealer_blackjack = (dealer_sum == 21 and len(self.dealer_hand) == 2)

        # Determine outcome and reward
        if player_sum > 21:
            return -1 # Player bust (this should ideally be caught earlier, but included for robustness)
        elif dealer_sum > 21:
            return 1 # Dealer bust, player wins
        elif player_blackjack and not dealer_blackjack:
            return self.blackjack_payout # Player wins with blackjack
        elif dealer_blackjack and not player_blackjack:
            return -1 # Dealer wins with blackjack
        elif player_sum > dealer_sum:
            return 1 # Player wins by higher score
        elif player_sum < dealer_sum:
            return -1 # Dealer wins by higher score
        else:
            return 0 # Push (tie)

    def render(self):
        """
        Renders the current state of the game to the console in human-readable format.
        """
        if self.render_mode == 'human':
            print("\n--- Blackjack Game ---")
            # Display all player hands (especially useful for splits)
            for i, hand_obj in enumerate(self.player_hands):
                player_sum, usable_ace = self._update_hand_value(hand_obj.cards)
                status = ""
                if hand_obj.stood: status += " (Stood)"
                if hand_obj.double_down: status += " (Doubled Down)"
                if player_sum > 21: status += " (Bust!)"
                print(f"Player Hand {i+1}: {[str(card) for card in hand_obj.cards]} (Sum: {player_sum}, Usable Ace: {usable_ace}){status}")
            
            # Display dealer's hand (showing one card initially, then all)
            dealer_sum, _ = self._update_hand_value(self.dealer_hand)
            # Show hole card as '??' if game is not over and it's the second card
            dealer_cards_str = [str(self.dealer_hand[0]), '??'] if len(self.dealer_hand) == 2 and self.current_hand_index < len(self.player_hands) else [str(card) for card in self.dealer_hand]
            print(f"Dealer Hand: {dealer_cards_str} (Showing: {self.dealer_hand[0].value}, Total: {dealer_sum if len(self.dealer_hand) > 1 else '??'})")
            print(f"Running Count: {self.running_count}")
            print("----------------------")
        else:
            # No rendering for other modes
            pass

    def close(self):
        """
        Performs any necessary cleanup (not applicable for this simple env).
        """
        pass
