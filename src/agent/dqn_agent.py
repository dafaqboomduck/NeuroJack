# blackjack_rl/agent/dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import logging
from tqdm.auto import tqdm # Changed import to tqdm.auto for better compatibility
from typing import Tuple, Union, Optional # Import Optional for type hints

# Configure logging for the DQN agent
logger = logging.getLogger(__name__)
# The logging level will be set dynamically in __init__ based on the verbose parameter

# Assuming these are correctly defined elsewhere in your project structure
from src.model.q_model import build_q_model
from src.memory.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,
                 env, # Accept the environment object directly
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.99,
                 replay_buffer_capacity=10000,
                 target_update_freq=100,
                 train_freq=1,
                 verbose=1,
                 model_name="DQN",
                 q_net_model: Optional[keras.Model] = None): # New parameter for a custom Q-network model

        # Dynamically derive state_size, num_actions, num_decks, and use_card_count from the environment
        self.state_size = env.state_size
        self.num_actions = env.num_actions
        self.num_decks = env.num_decks
        self.use_card_count = env.count_cards

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.replay_buffer_capacity = replay_buffer_capacity
        self.train_freq = train_freq
        self.verbose = bool(verbose)
        self.model_name = model_name

        # Set logger level based on verbose
        logger.setLevel(logging.INFO if self.verbose else logging.CRITICAL)

        # Build Q-networks based on the provided model or a default one
        if q_net_model is None:
            # Use the default model if none is provided
            self.q_net = build_q_model(input_shape=(self.state_size,), num_actions=self.num_actions)
        else:
            # Use the custom model provided by the user
            self.q_net = q_net_model
            logger.info("Using custom Q-network model provided by user.")

        # Create a separate target Q-network with the same architecture and initial weights
        self.target_q_net = keras.models.clone_model(self.q_net)
        self.target_q_net.set_weights(self.q_net.get_weights())

        # Optimizer and Loss
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)

    def _preprocess_state(self, state: tuple) -> np.ndarray:
        """
        Convert state tuple into a normalized float32 vector.
        This method now uses self.use_card_count to determine expected state size
        and also handles the 'true_count' if card counting is enabled.
        """
        player_sum, dealer_card, usable_ace = 0, 0, 0
        running_count = 0
        true_count = 0

        if self.use_card_count:
            if len(state) != 5:
                raise ValueError(f"Expected 5 values in state for card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace, running_count, true_count = state
        else:
            if len(state) != 3:
                raise ValueError(f"Expected 3 values in state without card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace = state

        # Normalization values
        player_sum_norm = (player_sum - 2) / (22 - 2)
        dealer_card_norm = (dealer_card - 1) / (11 - 1)
        usable_ace_norm = float(usable_ace)

        state_vector = [player_sum_norm, dealer_card_norm, usable_ace_norm]

        if self.use_card_count:
            max_running_count_abs = self.num_decks * 10
            running_count_norm = running_count / max_running_count_abs
            
            # true_count_norm = true_count / max_true_count_abs is not needed. The user needs to normalize based on the number of cards in the deck
            # The count depends on the number of decks being played
            true_count_norm = true_count / self.num_decks
            
            state_vector.append(running_count_norm)
            state_vector.append(true_count_norm)

        if len(state_vector) != self.state_size:
            raise ValueError(f"Mismatch in processed state vector length ({len(state_vector)}) "
                             f"and agent's expected state_size ({self.state_size}). "
                             "Check DQNAgent initialization and _preprocess_state logic.")

        return np.array(state_vector, dtype=np.float32)

    def choose_action(self, state: np.ndarray, available_actions: list) -> int:
        """
        Chooses an action using an epsilon-greedy policy, respecting available_actions.
        """
        valid_action_indices = [i for i, valid in enumerate(available_actions) if valid]

        if not valid_action_indices:
            logger.warning("No valid actions provided. Defaulting to action 0 (Stand).")
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_action_indices)
        else:
            state_input = np.expand_dims(state, axis=0)
            q_values = self.q_net(state_input)[0].numpy()

            masked_q_values = np.array(q_values)
            for i in range(len(masked_q_values)):
                if i not in valid_action_indices:
                    masked_q_values[i] = -np.inf

            if np.all(masked_q_values == -np.inf):
                logger.warning("All valid actions have -inf Q-value. Falling back to random valid action.")
                return random.choice(valid_action_indices)

            return np.argmax(masked_q_values)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Stores an experience in the replay buffer.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    @tf.function
    def _train_step(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, next_states: tf.Tensor, dones: tf.Tensor) -> tf.Tensor:
        """
        Performs a single training step for the Q-network using standard DQN update.
        """
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        next_q_values_target = self.target_q_net(next_states)
        max_next_q = tf.reduce_max(next_q_values_target, axis=1)

        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            q_values = self.q_net(states)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
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

    def learn(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        self._train_step(states, actions, rewards, next_states, dones)

    def fit(self, env, num_episodes: int, batch_size: int, log_interval: int) -> list:
        """
        Trains the DQN agent in the given environment.
        """
        rewards_history = []
        target_update_counter = 0
        steps_in_interval = 0
        current_interval_rewards = []
        batch_num = 0
        global_step_counter = 0 # Counter for total steps to control learn frequency

        logger.info(f"Starting {self.model_name} training for {num_episodes} episodes...")

        pbar_batch = tqdm(total=log_interval, desc=f"Batch {batch_num + 1}/{num_episodes // log_interval}",
                          unit=" episode", leave=True, dynamic_ncols=True, disable=not self.verbose)

        try:
            for episode in range(num_episodes):
                obs, info = env.reset()
                state = self._preprocess_state(obs)
                done = False
                total_reward = 0

                while not done:
                    available_actions = [True, True]

                    if env.allow_doubling:
                        available_actions.append(info.get('can_double', False))
                    if env.allow_splitting:
                        available_actions.append(info.get('can_split', False))

                    while len(available_actions) < self.num_actions:
                        available_actions.append(False)
                
                    action = self.choose_action(state, available_actions)

                    next_obs, reward, done, info = env.step(action)
                    next_state = self._preprocess_state(next_obs)

                    self.remember(state, action, reward, next_state, done)

                    global_step_counter += 1 # INCREMENT GLOBAL STEP COUNTER
                    if global_step_counter % self.train_freq == 0: # CONDITION TO CALL LEARN
                        self.learn(batch_size)

                    state = next_state
                    total_reward += reward
                    steps_in_interval += 1

                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

                target_update_counter += 1
                if target_update_counter >= self.target_update_freq:
                    self.update_target_model()
                    target_update_counter = 0

                rewards_history.append(total_reward)
                current_interval_rewards.append(total_reward)

                pbar_batch.update(1)
                pbar_batch.set_postfix({
                    'AvgR': f"{np.mean(current_interval_rewards):.2f}" if current_interval_rewards else 'N/A',
                    'Eps': f"{self.epsilon:.3f}",
                    'Buf': f"{len(self.replay_buffer)}",
                    'Steps/Int': f"{steps_in_interval}"
                })

                if (episode + 1) % log_interval == 0:
                    batch_num += 1
                    avg_reward_interval = np.mean(current_interval_rewards)

                    logger.info(
                        f"\nEpisode Batch {episode + 1}/{num_episodes}, "
                        f"Avg Reward (last {log_interval}): {avg_reward_interval:.4f}"
                    )
                    steps_in_interval = 0
                    current_interval_rewards = []

                    if (episode + 1) < num_episodes:
                        pbar_batch.reset(total=log_interval)
                        pbar_batch.set_description(f"Batch {batch_num + 1}/{num_episodes // log_interval}")
                        pbar_batch.refresh()

        finally:
            pbar_batch.close()

        logger.info(f"{self.model_name} training complete.")
        return rewards_history

    def evaluate(self, environment, num_eval_episodes: int, show_win_loss_rates: bool = False) -> Union[float, Tuple[float, float, float, float]]:
        """
        Evaluates the agent's performance in the given environment.
        The primary metric is average reward. Optionally shows win/loss/push rates.

        Args:
            environment: The Blackjack environment to evaluate on.
            num_eval_episodes (int): The number of episodes to run for evaluation.
            show_win_loss_rates (bool): If True, also returns and logs win/loss/push rates.

        Returns:
            Union[float, Tuple[float, float, float, float]]:
                If show_win_loss_rates is False, returns the average reward.
                If show_win_loss_rates is True, returns (average_reward, win_rate, push_rate, loss_rate).
        """
        logger.info(f"Starting evaluation for {num_eval_episodes} episodes...")
        total_rewards = []
        wins = 0
        pushes = 0
        losses = 0

        original_epsilon = self.epsilon
        self.epsilon = 0.0 # Agent acts greedily during evaluation

        with tqdm(range(num_eval_episodes), desc=f"{self.model_name} Evaluation", unit="episode",
                  leave=True, dynamic_ncols=True, disable=not self.verbose) as pbar_eval:
            for episode in pbar_eval:
                raw_state, info = environment.reset()
                state = self._preprocess_state(raw_state)
                done = False
                episode_reward = 0

                while not done:
                    available_actions = [True, True]
                    if environment.allow_doubling:
                        available_actions.append(info.get('can_double', False))
                    if environment.allow_splitting:
                        available_actions.append(info.get('can_split', False))
                    while len(available_actions) < self.num_actions:
                        available_actions.append(False)

                    action = self.choose_action(state, available_actions)
                    next_state_raw, reward, done, info = environment.step(action)
                    state = self._preprocess_state(next_state_raw)
                    episode_reward += reward # Accumulate reward for the episode

                total_rewards.append(episode_reward)

                if episode_reward > 0:
                    wins += 1
                elif episode_reward < 0:
                    losses += 1
                else:
                    pushes += 1

                # Update evaluation progress bar description
                postfix_dict = {'AvgR': f"{np.mean(total_rewards):.2f}"}
                if show_win_loss_rates:
                    postfix_dict['Wins'] = f"{wins}"
                    postfix_dict['Pushes'] = f"{pushes}"
                    postfix_dict['Losses'] = f"{losses}"
                pbar_eval.set_postfix(postfix_dict)

        self.epsilon = original_epsilon # Restore original epsilon

        total_episodes = len(total_rewards)
        average_reward = np.mean(total_rewards) if total_episodes > 0 else 0.0
        win_rate = wins / total_episodes if total_episodes > 0 else 0
        push_rate = pushes / total_episodes if total_episodes > 0 else 0
        loss_rate = losses / total_episodes if total_episodes > 0 else 0


        logger.info("\n--- Evaluation Results ---")
        logger.info(f"Total Episodes: {total_episodes}")
        logger.info(f"Average Reward: {average_reward:.4f}")
        if show_win_loss_rates:
            logger.info(f"Wins: {wins} ({win_rate:.2%})")
            logger.info(f"Pushes: {pushes} ({push_rate:.2%})")
            logger.info(f"Losses: {loss_rate:.2%})") # Corrected typo: should be losses not loss_rate
        logger.info("--------------------------")

        if show_win_loss_rates:
            return average_reward, win_rate, push_rate, loss_rate
        else:
            return average_reward

    def save_weights(self, path: str = None):
        """
        Saves the Q-network weights.
        """
        if path is None:
            logger.error("Error: A path must be provided to save weights.")
            return
        self.q_net.save_weights(path)
        logger.info(f"{self.model_name} model weights saved to {path}")

    def load_weights(self, path: str = None):
        """
        Loads Q-network weights.
        """
        if path is None:
            logger.error("Error: A path must be provided to load weights.")
            return
        try:
            self.q_net.load_weights(path)
            self.target_q_net.set_weights(self.q_net.get_weights())
            logger.info(f"{self.model_name} model weights loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model weights from {path}: {e}")
