# blackjack_rl/agent/double_dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.agent.dqn_agent import DQNAgent
from src.config import settings

class DoubleDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the model save path for Double DQN
        self.model_save_path = None
        print("DoubleDQNAgent initialized.")

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        Performs a single training step for the Q-network using Double DQN update.
        """
        actions = tf.cast(actions, tf.int32)

        with tf.GradientTape() as tape:
            # Get Q-values for the current states from the main Q-network
            q_values = self.q_net(states)
            action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
            predicted_q_values = tf.gather_nd(q_values, action_indices)

            # Double DQN update:
            # 1. Use main Q-network to select the best action for the next state
            next_q_values_main = self.q_net(next_states)
            selected_next_actions = tf.argmax(next_q_values_main, axis=1, output_type=tf.int32)

            # 2. Use target Q-network to evaluate the value of the selected action
            next_q_values_target = self.target_q_net(next_states)
            selected_next_action_indices = tf.stack([tf.range(len(selected_next_actions)), selected_next_actions], axis=1)
            max_next_q = tf.gather_nd(next_q_values_target, selected_next_action_indices)

            # Calculate target Q-values
            targets = rewards + (1 - dones) * self.gamma * max_next_q

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
        super().save_weights(path)

    def load_weights(self, path=None):
        """
        Loads Q-network weights for Double DQN.
        """
        super().load_weights(path)