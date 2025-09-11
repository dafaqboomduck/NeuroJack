# blackjack_rl/agent/double_dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from tqdm.auto import tqdm
from typing import Tuple, Union, Optional

from neurojack.agent import DQNAgent

# Configure logging for the Double DQN agent
# The logging level will be set dynamically in __init__ based on the verbose parameter
logger = logging.getLogger(__name__)


class DoubleDQNAgent(DQNAgent):
    """
    DoubleDQNAgent class, a subclass of DQNAgent that implements Double Deep Q-Learning.

    Double DQN addresses the problem of overestimation of Q-values that can occur
    in standard DQN. It decouples the action selection from the action evaluation.
    The main Q-network is used to select the greedy action, and the target Q-network
    is used to evaluate the Q-value of that selected action. This leads to more
    stable and reliable learning.

    This agent inherits all the core functionalities from the base DQNAgent class
    and only overrides the `_train_step` method, which contains the specific
    Double DQN update rule.

    Attributes:
        Inherits all attributes from the base DQNAgent class.
    
    Methods:
        __init__(self, *args, **kwargs): Initializes the DoubleDQNAgent.
        _train_step(self, states, actions, rewards, next_states, dones):
            Performs the Double DQN training update step.
        evaluate(self, environment, num_eval_episodes, show_win_loss_rates):
            Evaluates the agent's performance.
        save_weights(self, path): Saves the model weights.
        load_weights(self, path): Loads the model weights.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the DoubleDQNAgent.

        This constructor calls the parent class's `__init__` method, ensuring all
        the base DQN agent's components are set up correctly. It also sets the
        model name to "Double DQN" by default, which is used for logging and
        identification.
        """
        # Set a default model name specific to Double DQN if not provided
        kwargs.setdefault('model_name', "Double DQN")
        super().__init__(*args, **kwargs)
        logger.info(f"{self.model_name} initialized.")

    @tf.function
    def _train_step(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, next_states: tf.Tensor, dones: tf.Tensor) -> tf.Tensor:
        """
        Performs a single training step using the Double DQN update rule.

        This method is the core of the Double DQN algorithm. Unlike standard DQN,
        which uses the target network to find the maximum next Q-value, Double DQN
        uses the main Q-network to *select* the best next action and the target
        Q-network to *evaluate* the Q-value of that action. This prevents the
        overestimation of values that can occur when the same network is used for
        both selection and evaluation.

        The target Q-value calculation is defined by the following equation:
        "$Q_{target} = R_{t+1} + \\gamma * Q_{target}(S_{t+1}, \\underset{a}{\operatorname{argmax}} Q_{main}(S_{t+1}, a))$"
        where:
        - $R_{t+1}$ is the reward at the next step.
        - $\gamma$ is the discount factor.
        - $S_{t+1}$ is the next state.
        - $Q_{main}$ is the main Q-network.
        - $Q_{target}$ is the target Q-network.

        Args:
            states (tf.Tensor): A batch of states.
            actions (tf.Tensor): A batch of actions taken.
            rewards (tf.Tensor): A batch of rewards received.
            next_states (tf.Tensor): A batch of next states.
            dones (tf.Tensor): A batch of booleans indicating if the episode ended.

        Returns:
            tf.Tensor: The calculated loss for the training step.
        """
        # Ensure correct data types for TensorFlow operations
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        with tf.GradientTape() as tape:
            # 1. Get predicted Q-values from the main Q-network for the states and actions in the batch.
            q_values = self.q_net(states)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            predicted_q_values = tf.gather_nd(q_values, action_indices)

            # 2. Use the main Q-network to select the best action for the next state (action selection).
            next_q_values_main = self.q_net(next_states)
            selected_next_actions = tf.argmax(next_q_values_main, axis=1, output_type=tf.int32)

            # 3. Use the target Q-network to evaluate the value of the selected action (value evaluation).
            next_q_values_target = self.target_q_net(next_states)
            selected_next_action_indices = tf.stack([tf.range(tf.shape(selected_next_actions)[0], dtype=tf.int32), selected_next_actions], axis=1)
            max_next_q = tf.gather_nd(next_q_values_target, selected_next_action_indices)

            # 4. Calculate the target Q-values using the Bellman equation.
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q

            # 5. Calculate the loss between the predicted and target Q-values.
            loss = self.loss_fn(targets, predicted_q_values)

        # 6. Compute gradients and apply them to update the main Q-network.
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def evaluate(self, environment, num_eval_episodes: int, show_win_loss_rates: bool = False) -> Union[float, Tuple[float, float, float, float]]:
        """
        Evaluates the agent's performance in the given environment.

        This method overrides the parent's `evaluate` method to provide a dedicated
        docstring for this specific subclass. The implementation itself is
        inherited from the base class, as the evaluation process (using a greedy
        policy) is the same for both DQN and Double DQN.

        Args:
            environment: The Blackjack environment to evaluate on.
            num_eval_episodes (int): The number of episodes to run for evaluation.
            show_win_loss_rates (bool): If `True`, also returns and logs win/loss/push rates.

        Returns:
            Union[float, Tuple[float, float, float, float]]:
                If `show_win_loss_rates` is `False`, returns the average reward.
                If `show_win_loss_rates` is `True`, returns a tuple of
                `(average_reward, win_rate, push_rate, loss_rate)`.
        """
        return super().evaluate(environment, num_eval_episodes, show_win_loss_rates)

    def save_weights(self, path: str = None):
        """
        Saves the weights of the main Q-network.

        This method calls the `save_weights` method of the parent class (`DQNAgent`)
        to save the weights of the main Q-network. The saved file can be used later
        to load the trained model.
        
        Args:
            path (str): The file path to save the weights to.
        """
        super().save_weights(path)

    def load_weights(self, path: str = None):
        """
        Loads Q-network weights from a file into both the main and target networks.

        This method calls the `load_weights` method of the parent class (`DQNAgent`)
        to load a previously saved model. It ensures the weights are loaded into
        both the main and target networks, maintaining synchronization.

        Args:
            path (str): The file path to load the weights from.
        """
        super().load_weights(path)
