# blackjack_rl/agent/dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

from src.model.q_model import build_q_model
from src.memory.replay_buffer import ReplayBuffer
# Removed: from src.utils.helpers import preprocess_state # No longer needed, as it's a method now
from src.config import settings

class DQNAgent:
    def __init__(self,
                 state_size=settings.STATE_SIZE,
                 num_actions=settings.NUM_ACTIONS,
                 learning_rate=settings.LEARNING_RATE,
                 gamma=settings.GAMMA,
                 epsilon_start=settings.EPSILON_START,
                 epsilon_end=settings.EPSILON_END,
                 epsilon_decay=settings.EPSILON_DECAY,
                 replay_buffer_capacity=settings.REPLAY_BUFFER_CAPACITY,
                 target_update_freq=settings.TARGET_UPDATE_FREQ):

        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Build Q-networks
        self.q_net = build_q_model(input_shape=(state_size,), num_actions=num_actions)
        self.target_q_net = build_q_model(input_shape=(state_size,), num_actions=num_actions)
        self.target_q_net.set_weights(self.q_net.get_weights()) # Initialize target network to be same as q_net

        # Optimizer and Loss
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    def _preprocess_state(self, state, use_card_count=False):
        """
        Convert state tuple into a normalized float32 vector.

        Args:
            state: Tuple of environment state.
            use_card_count (bool): Whether to include running_count in the state.

        Returns:
            np.array: Normalized state vector.
        """
        if use_card_count:
            if len(state) != 4:
                raise ValueError(f"Expected 4 values in state, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace, running_count = state
        else:
            if len(state) != 3:
                raise ValueError(f"Expected 3 values in state, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace = state

        # Normalize
        state_vector = [
            (player_sum - 12) / (21 - 12),        # Player sum normalized
            (dealer_card - 1) / (10 - 1),         # Dealer card normalized
            float(usable_ace)                     # Usable ace as float
        ]

        if use_card_count:
            max_running_count_abs = 20 * 6  # Adjust based on number of decks
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

    def fit(self, env, num_episodes=settings.NUM_EPISODES, batch_size=settings.BATCH_SIZE,
        log_interval=settings.TARGET_UPDATE_FREQ):
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
            state = self._preprocess_state(obs) # Use self._preprocess_state
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state_raw, reward, done, _ = env.step(action)
                next_state = self._preprocess_state(next_state_raw) # Use self._preprocess_state

                self.remember(state, action, reward, next_state, done)
                self.learn(batch_size)

                state = next_state
                total_reward += reward

                # The batch sampling and training is already handled by self.learn()
                # This block is redundant if self.learn() is called every step
                # if len(self.replay_buffer) > batch_size:
                #     states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                #         self.replay_buffer.sample(batch_size)
                #     self._train_step(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

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
            state = self._preprocess_state(raw_state) # Use self._preprocess_state
            done = False

            while not done:
                action = self.choose_action(state)  # Agent acts greedily based on Q-values
                next_state_raw, reward, done, _ = environment.step(action)
                state = self._preprocess_state(next_state_raw) # Use self._preprocess_state

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