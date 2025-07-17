# blackjack_rl/agent/double_dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging # Import logging

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