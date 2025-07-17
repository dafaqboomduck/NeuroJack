# blackjack_rl/agent/dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Assuming these are correctly defined elsewhere in your project structure
from src.model.q_model import build_q_model
from src.memory.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,
                 state_size=3,
                 num_actions=2,
                 num_decks=1, # This parameter is primarily for normalization in preprocess_state
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 replay_buffer_capacity=10000,
                 target_update_freq=100,
                 use_card_count=False):

        self.state_size = state_size
        self.num_actions = num_actions
        self.num_decks = num_decks # Used for max_running_count_abs calculation
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.use_card_count = use_card_count # Store this flag

        # Build Q-networks with the dynamically determined state_size
        self.q_net = build_q_model(input_shape=(self.state_size,), num_actions=self.num_actions)
        self.target_q_net = build_q_model(input_shape=(self.state_size,), num_actions=self.num_actions)
        self.target_q_net.set_weights(self.q_net.get_weights()) # Initialize target network to be same as q_net

        # Optimizer and Loss
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    def _preprocess_state(self, state):
        """
        Convert state tuple into a normalized float32 vector.
        This method now uses self.use_card_count to determine expected state size
        and also handles the 'true_count' if card counting is enabled.
        """
        player_sum, dealer_card, usable_ace = 0, 0, 0
        running_count = 0
        true_count = 0 # Initialize true_count

        if self.use_card_count:
            # The environment now returns 5 values:
            # (player_current_sum, dealer_card_showing, usable_ace, running_count, true_count)
            if len(state) != 5: # <-- CORRECTED FROM 4 TO 5
                raise ValueError(f"Expected 5 values in state for card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace, running_count, true_count = state # <-- ADDED true_count HERE
        else:
            if len(state) != 3:
                raise ValueError(f"Expected 3 values in state without card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace = state
            # running_count and true_count remain 0 if not using card counting

        # Normalization values (consider making these configurable or more robust)
        # Max player sum can be 21 for a non-bust hand. If bust, it can be higher (e.g., 22 for 11+11).
        # Normalizing by 22.0 ensures values are between 0 and ~1.
        player_sum_norm = (player_sum - 2) / (22 - 2) # Assuming min player sum is 2 (e.g., 2,2)
        dealer_card_norm = (dealer_card - 1) / (11 - 1) # Dealer card value ranges from 2-11. Ace is 11. Min 1 is for placeholder.
        usable_ace_norm = float(usable_ace) # Already 0 or 1

        state_vector = [player_sum_norm, dealer_card_norm, usable_ace_norm]

        if self.use_card_count:
            # Normalizing running_count and true_count
            # The running count can fluctuate significantly. A rough maximum for 6 decks might be around +20 to +30
            # or -20 to -30. Normalizing by something like (num_decks * 10) provides a decent range.
            max_running_count_abs = self.num_decks * 10 # Example: For 6 decks, max ~60
            running_count_norm = running_count / max_running_count_abs

            # True count can also vary. A common range for true count might be -10 to +10.
            # Normalizing by a fixed value like 10.0 or (num_decks * 2)
            max_true_count_abs = self.num_decks * 2 # Example: For 6 decks, max ~12
            true_count_norm = true_count / max_true_count_abs

            state_vector.append(running_count_norm) #
            state_vector.append(true_count_norm) #

        # Ensure the state_size attribute of the agent correctly matches the length of the processed_state vector.
        # This is CRITICAL for the neural network input shape.
        if len(state_vector) != self.state_size:
            raise ValueError(f"Mismatch in processed state vector length ({len(state_vector)}) "
                             f"and agent's expected state_size ({self.state_size}). "
                             "Check DQNAgent initialization and _preprocess_state logic.")

        return np.array(state_vector, dtype=np.float32)

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        # Ensure state is correctly shaped for the model
        state_input = np.expand_dims(state, axis=0)

        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.q_net(state_input)
        return np.argmax(q_values[0].numpy())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        # Double DQN update: Use online Q-net to select action, target Q-net to evaluate action
        next_q_values_online = self.q_net(next_states)
        best_next_actions = tf.argmax(next_q_values_online, axis=1) # This is likely int64 by default

        next_q_values_target = self.target_q_net(next_states)

        # Gather Q-values from target net using actions chosen by online net
        # Fix: Cast best_next_actions to tf.int32 to match tf.range's output type
        target_q_at_best_action = tf.gather_nd(
            next_q_values_target,
            tf.stack([tf.range(len(best_next_actions), dtype=tf.int32),  # Explicitly cast tf.range output
                      tf.cast(best_next_actions, tf.int32)],  # Explicitly cast best_next_actions
                     axis=1)
        )

        targets = rewards + (1.0 - dones) * self.gamma * target_q_at_best_action

        with tf.GradientTape() as tape:
            q_values = self.q_net(states)
            action_indices = tf.stack([tf.range(len(actions), dtype=tf.int32), actions], axis=1) # Also apply here for consistency if actions might vary
            predicted_q_values = tf.gather_nd(q_values, action_indices)

            loss = self.loss_fn(targets, predicted_q_values)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def update_target_model(self):
        """
        Updates the target Q-network weights from the main Q-network.
        This is a hard update.
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        self._train_step(states, actions, rewards, next_states, dones)

    def fit(self, env, num_episodes, batch_size, log_interval):
        """
        Trains the DQN agent in the given environment.

        Args:
            env: The environment to train in (your CustomBlackjackEnv instance).
            num_episodes: Total number of episodes to train for.
            batch_size: Size of the batch to sample from the replay buffer.
            log_interval: How often to print training progress (in episodes).

        Returns:
            A list of rewards obtained per episode during training.
        """
        rewards_history = []
        target_update_counter = 0 # Counter for target network updates

        for episode in range(num_episodes):
            obs, info = env.reset() # Get info dictionary during reset
            state = self._preprocess_state(obs)
            done = False
            total_reward = 0

            while not done:
                # Get available actions from the environment's info dictionary
                # This assumes env.step and env.reset provide 'can_double' and 'can_split' in 'info'
                available_actions = [True, True] # Stand (0) and Hit (1) are generally always available

                # Dynamically append based on environment's capabilities and current state info
                if env.allow_doubling:
                    available_actions.append(info.get('can_double', False))
                if env.allow_splitting:
                    available_actions.append(info.get('can_split', False))

                # Pad available_actions list with False if it's shorter than self.num_actions
                # This can happen if the agent was initialized with more actions than the current env allows,
                # or if some actions are globally disabled in the env.
                while len(available_actions) < self.num_actions:
                    available_actions.append(False)


                action = self.choose_action_with_mask(state, available_actions) # Use the new method

                next_obs, reward, done, info = env.step(action)
                next_state = self._preprocess_state(next_obs)

                self.remember(state, action, reward, next_state, done)
                self.learn(batch_size) # Agent learns at each step

                state = next_state
                total_reward += reward

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Update target network based on frequency
            target_update_counter += 1
            if target_update_counter >= self.target_update_freq:
                self.update_target_model()
                target_update_counter = 0 # Reset counter

            rewards_history.append(total_reward)

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(rewards_history[-log_interval:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last {log_interval}): {avg_reward:.4f}, Epsilon: {self.epsilon:.3f}, Replay Buffer Size: {len(self.replay_buffer)}")

        return rewards_history

    def choose_action_with_mask(self, state, available_actions):
        """
        Chooses an action using an epsilon-greedy policy, respecting available_actions.
        """
        valid_action_indices = [i for i, valid in enumerate(available_actions) if valid]

        if not valid_action_indices:
            # Fallback if no valid actions are provided (should ideally not happen)
            # You might want to log a warning here
            print("Warning: No valid actions provided. Defaulting to action 0 (Stand).")
            return 0 # Default to stand or first action

        if random.random() < self.epsilon:
            # Explore: choose a random valid action
            return random.choice(valid_action_indices)
        else:
            # Exploit: choose the best valid action from the model's predictions
            state_input = np.expand_dims(state, axis=0)
            q_values = self.q_net(state_input)[0].numpy()

            # Mask out invalid actions by setting their Q-values to a very low number
            masked_q_values = np.array(q_values)
            for i in range(len(masked_q_values)):
                if i not in valid_action_indices:
                    masked_q_values[i] = -np.inf # Ensures these are not chosen

            # If all valid actions somehow have -inf (e.g., due to previous error), fall back
            if np.all(masked_q_values == -np.inf):
                return random.choice(valid_action_indices) # Fallback to random valid action

            return np.argmax(masked_q_values)

    def evaluate(self, environment, num_eval_episodes):
        """
        Evaluates the agent's performance in the given environment.

        Args:
            environment: The environment to evaluate in (your CustomBlackjackEnv instance).
            num_eval_episodes: The number of episodes to run for evaluation.

        Returns:
            tuple: (win_rate, push_rate, loss_rate)
        """
        print(f"Starting evaluation for {num_eval_episodes} episodes...")
        wins = 0
        pushes = 0
        losses = 0

        # Temporarily set epsilon to 0 for evaluation (no exploration)
        original_epsilon = self.epsilon
        self.epsilon = 0.0 # Agent acts greedily during evaluation

        for episode in range(num_eval_episodes):
            raw_state, info = environment.reset() # Get info for available actions
            state = self._preprocess_state(raw_state)
            done = False

            while not done:
                # Get available actions for evaluation
                available_actions = [True, True]
                if environment.allow_doubling:
                    available_actions.append(info.get('can_double', False))
                if environment.allow_splitting:
                    available_actions.append(info.get('can_split', False))
                while len(available_actions) < self.num_actions:
                    available_actions.append(False)

                action = self.choose_action_with_mask(state, available_actions) # Use the masked action choice
                next_state_raw, reward, done, info = environment.step(action) # Pass info along
                state = self._preprocess_state(next_state_raw)

            # Important: The reward from env.step is the total reward for all hands in the episode.
            # You might need to adjust your reward tracking if you want per-hand win/loss stats.
            # Assuming current `reward` is the final sum of rewards for the episode.
            if reward > 0: # A reward > 0 implies a win (including blackjack payouts)
                wins += 1
            elif reward < 0: # A reward < 0 implies a loss (including busts)
                losses += 1
            else: # reward == 0 implies a push
                pushes += 1

            if (episode + 1) % 1000 == 0:
                print(f"  Evaluation episode {episode + 1}/{num_eval_episodes}")

        self.epsilon = original_epsilon # Restore original epsilon

        total_episodes = wins + pushes + losses
        # Handle division by zero if total_episodes is 0
        win_rate = wins / total_episodes if total_episodes > 0 else 0
        push_rate = pushes / total_episodes if total_episodes > 0 else 0
        loss_rate = losses / total_episodes if total_episodes > 0 else 0


        print("\n--- Evaluation Results ---")
        print(f"Total Episodes: {total_episodes}")
        print(f"Wins: {wins} ({win_rate:.2%})")
        print(f"Pushes: {pushes} ({push_rate:.2%})")
        print(f"Losses: {losses} ({loss_rate:.2%})")
        print("--------------------------")
        return win_rate, push_rate, loss_rate

    def save_weights(self, path=None):
        """
        Saves the Q-network weights.
        """
        if path is None:
            print("Error: A path must be provided to save weights.")
            return
        self.q_net.save_weights(path)
        print(f"DQN model weights saved to {path}")

    def load_weights(self, path=None):
        """
        Loads Q-network weights.
        """
        if path is None:
            print("Error: A path must be provided to load weights.")
            return
        try:
            self.q_net.load_weights(path)
            self.target_q_net.set_weights(self.q_net.get_weights())
            print(f"DQN model weights loaded from {path}")
        except Exception as e:
            print(f"Error loading DQN model weights from {path}: {e}")