# blackjack_rl/agent/double_dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging # Import logging
from tqdm.auto import tqdm # Changed import to tqdm.auto for better compatibility
from typing import Tuple, Union # Import Union and Tuple for type hints

from src.agent.dqn_agent import DQNAgent

# Configure logging for the Double DQN agent
# This logger will inherit settings from the root logger or be configured by DQNAgent's verbose
logger = logging.getLogger(__name__)
# It's generally good practice to set a default level here too,
# though DQNAgent's init might override it for the instance.
logger.setLevel(logging.INFO) # Default for this specific logger

class DoubleDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        # Pass model_name to the superclass constructor
        # If 'model_name' is not already in kwargs, set it to "Double DQN"
        kwargs.setdefault('model_name', "Double DQN")
        super().__init__(*args, **kwargs)
        # Override the model save path for Double DQN (if needed, though it's not used in parent)
        self.model_save_path = None # This line seems to be a remnant, as parent doesn't use it.
        logger.info(f"{self.model_name} initialized.") # Changed print to logger.info

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        Performs a single training step for the Q-network using Double DQN update.
        """
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32) # Ensure rewards are float32
        dones = tf.cast(dones, tf.float32) # Ensure dones are float32

        with tf.GradientTape() as tape:
            # Get Q-values for the current states from the main Q-network
            q_values = self.q_net(states)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1) # Fixed dtype mismatch here too
            predicted_q_values = tf.gather_nd(q_values, action_indices)

            # Double DQN update:
            # 1. Use main Q-network to select the best action for the next state
            next_q_values_main = self.q_net(next_states)
            selected_next_actions = tf.argmax(next_q_values_main, axis=1, output_type=tf.int32) # Ensure output_type is int32

            # 2. Use target Q-network to evaluate the value of the selected action
            next_q_values_target = self.target_q_net(next_states)
            # Fixed dtype mismatch here too
            selected_next_action_indices = tf.stack([tf.range(tf.shape(selected_next_actions)[0], dtype=tf.int32), selected_next_actions], axis=1)
            max_next_q = tf.gather_nd(next_q_values_target, selected_next_action_indices)

            # Calculate target Q-values
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q

            # Calculate loss
            loss = self.loss_fn(targets, predicted_q_values)

        # Compute gradients and apply them
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

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
            logger.info(f"Losses: {loss_rate:.2%})")
        logger.info("--------------------------")

        if show_win_loss_rates:
            return average_reward, win_rate, push_rate, loss_rate
        else:
            return average_reward

    def save_weights(self, path=None):
        """
        Saves the Q-network weights for Double DQN.
        """
        super().save_weights(path) # This will now use self.model_name from parent

    def load_weights(self, path=None):
        """
        Loads Q-network weights for Double DQN.
        """
        super().load_weights(path) # This will now use self.model_name from parent
