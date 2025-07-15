import numpy as np
import random
# Assuming Card, Deck, PlayerHand are imported from their respective files
from src.env.playerhand import PlayerHand
from src.env.deck import Deck

class CustomBlackjackEnv:
    """
    Custom Blackjack Environment with additional rules and card counting,
    without relying on Gymnasium.

    Observation Space:
    Tuple: (player_current_sum, dealer_card_showing, usable_ace[, running_count])
    - player_current_sum: Sum of player's current hand (int, 2–22).
    - dealer_card_showing: Value of dealer's visible card (int, 1–11, Ace=11).
    - usable_ace: Whether player has a usable ace (int, 0 or 1).
    - running_count: Current Hi-Lo running count (if count_cards=True)

    Action Space:
    0: Stand
    1: Hit
    2: Double Down (if allowed)
    3: Split (if allowed)
    """

    def __init__(self, render_mode=None, num_decks=6, blackjack_payout=1.5,
                 allow_doubling=True, allow_splitting=True, count_cards=True, seed=None):
        self.num_decks = num_decks
        self.blackjack_payout = blackjack_payout
        self.allow_doubling = allow_doubling
        self.allow_splitting = allow_splitting
        self.count_cards = count_cards # This is the crucial flag
        self.seed = seed
        self.render_mode = render_mode

        self.observation_description = (
            "(player_current_sum, dealer_card_showing, usable_ace"
            + (", running_count)" if self.count_cards else ")")
        )
        actions = ["0: Stand", "1: Hit"]
        if self.allow_doubling:
            actions.append("2: Double Down")
        if self.allow_splitting:
            actions.append("3: Split")
        self.action_description = ", ".join(actions)

        self.deck = Deck(self.num_decks, seed=self.seed)
        self.player_hands = []
        self.dealer_hand = []
        self.current_hand_index = 0
        self.running_count = 0

        self.reset()

    @property
    def state_size(self):
        # This property dynamically returns the correct state size
        return 4 if self.count_cards else 3

    @property
    def num_actions(self):
        # Always 0 and 1 (Stand, Hit), and conditionally Double Down and Split
        return 2 + int(self.allow_doubling) + int(self.allow_splitting)

    def _update_hand_value(self, hand_cards):
        hand_sum = 0
        num_aces = 0
        for card in hand_cards:
            if card.rank == 'A':
                num_aces += 1
            hand_sum += card.value

        usable_ace = False
        if num_aces > 0 and hand_sum <= 21:
            usable_ace = True

        while hand_sum > 21 and num_aces > 0:
            hand_sum -= 10
            num_aces -= 1
            usable_ace = False

        return hand_sum, usable_ace

    def _deal_card(self, hand_obj_or_list, is_player=True, face_up=True):
        card = self.deck.deal_card()
        if isinstance(hand_obj_or_list, PlayerHand):
            hand_obj_or_list.add_card(card)
        else:
            hand_obj_or_list.append(card)

        if self.count_cards and face_up:
            self.running_count += card.count_value
        return card

    def _get_obs(self):
        # Return observation based on whether card counting is enabled
        if not (0 <= self.current_hand_index < len(self.player_hands)):
            # Return appropriate zero observation for terminal state
            return (0, 0, 0, 0) if self.count_cards else (0, 0, 0)

        current_player_hand_cards = self.player_hands[self.current_hand_index].cards
        player_sum, usable_ace = self._update_hand_value(current_player_hand_cards)
        dealer_showing_value = self.dealer_hand[0].value

        obs = (player_sum, dealer_showing_value, int(usable_ace))
        if self.count_cards:
            obs += (self.running_count,)
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
            random.seed(seed)
        self.deck = Deck(self.num_decks, seed=self.seed)

        self.player_hands = [PlayerHand()]
        self.dealer_hand = []
        self.current_hand_index = 0
        self.running_count = 0

        self._deal_card(self.player_hands[0], is_player=True, face_up=True)
        self._deal_card(self.dealer_hand, is_player=False, face_up=True)
        self._deal_card(self.player_hands[0], is_player=True, face_up=True)
        self._deal_card(self.dealer_hand, is_player=False, face_up=False)

        if self.count_cards:
            self.running_count = 0
            self.running_count += self.player_hands[0].cards[0].count_value
            self.running_count += self.player_hands[0].cards[1].count_value
            self.running_count += self.dealer_hand[0].count_value

        player_sum, _ = self._update_hand_value(self.player_hands[0].cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand)

        done = False
        reward = 0
        info = {"can_double": False, "can_split": False}

        player_blackjack = (player_sum == 21 and len(self.player_hands[0].cards) == 2)
        dealer_blackjack = (dealer_sum == 21 and len(self.dealer_hand) == 2)

        if player_blackjack and dealer_blackjack:
            reward = 0
            done = True
        elif player_blackjack:
            reward = self.blackjack_payout
            done = True
        elif dealer_blackjack:
            reward = -1
            done = True

        if not done:
            if self.allow_doubling:
                info["can_double"] = True
            if self.allow_splitting and self.player_hands[0].cards[0].rank == self.player_hands[0].cards[1].rank:
                info["can_split"] = True

        if done:
            self.player_hands[0].reward = reward

        observation = self._get_obs()
        if self.render_mode == 'human':
            self.render()
        return observation, info

    def step(self, action):
        reward = 0
        done = False
        info = {"can_double": False, "can_split": False}

        current_player_hand_obj = self.player_hands[self.current_hand_index]
        current_player_hand_cards = current_player_hand_obj.cards
        player_sum, usable_ace = self._update_hand_value(current_player_hand_cards)

        current_hand_resolved = False
        is_first_action = (len(current_player_hand_cards) == 2 and
                           not current_player_hand_obj.stood and
                           not current_player_hand_obj.double_down)

        if action == 1:  # Hit
            self._deal_card(current_player_hand_obj, is_player=True, face_up=True)
            player_sum, _ = self._update_hand_value(current_player_hand_obj.cards)
            if player_sum > 21:
                current_player_hand_obj.reward = -1
                if current_player_hand_obj.double_down:
                    current_player_hand_obj.reward *= 2
                current_hand_resolved = True
            else:
                current_player_hand_obj.stood = True
        elif action == 0:  # Stand
            current_player_hand_obj.stood = True
            current_hand_resolved = True
        elif action == 2 and self.allow_doubling and is_first_action:
            self._deal_card(current_player_hand_obj, is_player=True, face_up=True)
            player_sum, _ = self._update_hand_value(current_player_hand_obj.cards)
            current_player_hand_obj.double_down = True
            current_player_hand_obj.stood = True
            if player_sum > 21:
                current_player_hand_obj.reward = -1
                current_player_hand_obj.reward *= 2
            current_hand_resolved = True
        elif action == 3 and self.allow_splitting and is_first_action and \
             current_player_hand_cards[0].rank == current_player_hand_cards[1].rank:
            card1, card2 = current_player_hand_cards
            current_player_hand_obj.cards = [card1]
            new_hand = PlayerHand(cards=[card2])
            self.player_hands.insert(self.current_hand_index + 1, new_hand)

            self._deal_card(current_player_hand_obj, is_player=True, face_up=True)
            self._deal_card(new_hand, is_player=True, face_up=True)
        else:
            current_player_hand_obj.reward = -0.75
            current_player_hand_obj.stood = True
            current_hand_resolved = True

        if current_hand_resolved:
            done = self._advance_to_next_hand_or_resolve_game()

        # Determine the next_observation based on self.count_cards
        next_observation = (0, 0, 0, 0) if self.count_cards else (0, 0, 0)
        if not done:
            next_observation = self._get_obs()
            current_hand = self.player_hands[self.current_hand_index]
            cards = current_hand.cards
            if self.allow_doubling and len(cards) == 2 and not current_hand.stood:
                info["can_double"] = True
            if self.allow_splitting and len(cards) == 2 and cards[0].rank == cards[1].rank and not current_hand.stood:
                info["can_split"] = True

        final_reward = sum(hand.reward for hand in self.player_hands) if done else 0

        if self.render_mode == 'human':
            self.render()

        return next_observation, final_reward, done, info

    def _advance_to_next_hand_or_resolve_game(self):
        self.current_hand_index += 1
        if self.current_hand_index < len(self.player_hands):
            return False
        self._dealer_plays()
        for hand in self.player_hands:
            if hand.reward == 0:
                hand.reward = self._calculate_reward(hand.cards)
                if hand.double_down:
                    hand.reward *= 2
        return True

    def _dealer_plays(self):
        if self.count_cards and len(self.dealer_hand) > 1:
            self.running_count += self.dealer_hand[1].count_value

        dealer_sum, _ = self._update_hand_value(self.dealer_hand)
        while dealer_sum < 17:
            self._deal_card(self.dealer_hand, is_player=False, face_up=True)
            dealer_sum, _ = self._update_hand_value(self.dealer_hand)

    def _calculate_reward(self, player_hand_cards):
        player_sum, _ = self._update_hand_value(player_hand_cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand)

        player_blackjack = (player_sum == 21 and len(player_hand_cards) == 2)
        dealer_blackjack = (dealer_sum == 21 and len(self.dealer_hand) == 2)

        if player_sum > 21:
            return -1
        elif dealer_sum > 21:
            return 1
        elif player_blackjack and not dealer_blackjack:
            return self.blackjack_payout
        elif dealer_blackjack and not player_blackjack:
            return -1
        elif player_sum > dealer_sum:
            return 1
        elif player_sum < dealer_sum:
            return -1
        else:
            return 0

    def render(self):
        if self.render_mode != 'human':
            return
        print("\n--- Blackjack Game ---")
        for i, hand in enumerate(self.player_hands):
            hand_sum, usable_ace = self._update_hand_value(hand.cards)
            status = []
            if hand.stood: status.append("Stood")
            if hand.double_down: status.append("Doubled Down")
            if hand_sum > 21: status.append("Bust")
            status_str = ", ".join(status)
            print(f"Player Hand {i+1}: {[str(c) for c in hand.cards]} (Sum: {hand_sum}, Usable Ace: {usable_ace}) [{status_str}]")
        dealer_sum, _ = self._update_hand_value(self.dealer_hand)
        if len(self.dealer_hand) == 2 and self.current_hand_index < len(self.player_hands):
            dealer_cards = [str(self.dealer_hand[0]), '??']
            dealer_total = '??'
        else:
            dealer_cards = [str(c) for c in self.dealer_hand]
            dealer_total = dealer_sum
        print(f"Dealer Hand: {dealer_cards} (Showing: {self.dealer_hand[0].value}, Total: {dealer_total})")
        print(f"Running Count: {self.running_count}")
        print("----------------------")

    def close(self):
        pass
