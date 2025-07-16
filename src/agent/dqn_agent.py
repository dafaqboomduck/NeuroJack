# blackjack_rl/agent/dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

from src.model.q_model import build_q_model
from src.memory.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,
                 state_size=3,
                 num_actions=2,
                 num_decks=1,
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 replay_buffer_capacity=10000,
                 target_update_freq=100,
                 use_card_count=False): # New parameter to indicate if card counting is used

        self.state_size = state_size
        self.num_actions = num_actions
        self.num_decks = num_decks
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
        This method now uses self.use_card_count to determine expected state size.
        """
        if self.use_card_count:
            if len(state) != 4:
                raise ValueError(f"Expected 4 values in state for card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace, running_count = state
        else:
            if len(state) != 3:
                # This is the line that was causing the error if self.use_card_count was True
                # but the state was actually 3 elements.
                # The logic here is correct if self.use_card_count is correctly False.
                raise ValueError(f"Expected 3 values in state without card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace = state
            running_count = 0 # Default if not using card count, to allow consistent state_vector construction

        # Normalize
        state_vector = [
            (player_sum - 12) / (21 - 12),       # Player sum normalized (assuming min 12, max 21 relevant for player)
            (dealer_card - 1) / (10 - 1),        # Normalized dealer card (1-10, Ace is 11, so 1-11 range for observation)
            float(usable_ace)                    # Binary feature
        ]

        if self.use_card_count:
            # Use settings.NUM_DECKS for consistency in normalization
            max_running_count_abs = self.num_decks * 20 # Max absolute count for NUM_DECKS (approx)
            state_vector.append(running_count / max_running_count_abs)

        return np.array(state_vector, dtype=np.float32)

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.q_net(np.expand_dims(state, axis=0))
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

        next_q_values = self.target_q_net(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            q_values = self.q_net(states)
            action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
            predicted_q_values = tf.gather_nd(q_values, action_indices)

            loss = self.loss_fn(targets, predicted_q_values)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def update_target_model(self):
        """
        Updates the target Q-network weights from the main Q-network.
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        self._train_step(states, actions, rewards, next_states, dones)

    def fit(self, env, num_episodes, batch_size,
        log_interval):
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

        for episode in range(num_episodes):
            obs, _ = env.reset()
            state = self._preprocess_state(obs) # No longer pass use_card_count here
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state_raw, reward, done, _ = env.step(action)
                next_state = self._preprocess_state(next_state_raw) # No longer pass use_card_count here

                self.remember(state, action, reward, next_state, done)
                self.learn(batch_size)

                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.update_target_model() # Update target network every episode

            rewards_history.append(total_reward)

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(rewards_history[-log_interval:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last {log_interval}): {avg_reward:.4f}, Epsilon: {self.epsilon:.3f}")

        return rewards_history

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
            raw_state, _ = environment.reset()
            state = self._preprocess_state(raw_state) # No longer pass use_card_count here
            done = False

            while not done:
                action = self.choose_action(state)  # Agent acts greedily based on Q-values
                next_state_raw, reward, done, _ = environment.step(action)
                state = self._preprocess_state(next_state_raw) # No longer pass use_card_count here

            if reward == 1:
                wins += 1
            elif reward == 0:
                pushes += 1
            else: # reward == -1
                losses += 1

            if (episode + 1) % 1000 == 0:
                print(f"  Evaluation episode {episode + 1}/{num_eval_episodes}")

        self.epsilon = original_epsilon # Restore original epsilon

        total_episodes = wins + pushes + losses
        win_rate = wins / total_episodes
        push_rate = pushes / total_episodes
        loss_rate = losses / total_episodes

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
        self.q_net.save_weights(path)
        print(f"DQN model weights saved to {path}")

    def load_weights(self, path=None):
        """
        Loads Q-network weights.
        """
        try:
            self.q_net.load_weights(path)
            self.target_q_net.set_weights(self.q_net.get_weights())
            print(f"DQN model weights loaded from {path}")
        except Exception as e:
            print(f"Error loading DQN model weights from {path}: {e}")

