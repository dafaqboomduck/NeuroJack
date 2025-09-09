# blackjack_rl/model/q_model.py
"""
This module defines the Q-network model for a reinforcement learning agent.
The Q-network is a neural network that approximates the Q-value function,
which estimates the expected future rewards for an agent's actions in a
given state.
"""
import tensorflow as tf
from tensorflow import keras

# Configure TensorFlow to use a GPU if available and allow memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_q_model(input_shape=(3,), num_actions=2):
    """
    Builds and returns a Keras Sequential model for the Q-network.

    The model consists of a simple feed-forward neural network with two hidden
    layers and a linear output layer. It is designed to take the game state
    as input and output the Q-values for each possible action.

    Args:
        input_shape (tuple): The shape of the input state. For Blackjack, this
                             is a tuple like (player_sum, dealer_card, usable_ace).
        num_actions (int): The number of possible actions the agent can take.
                           For Blackjack, this is typically 2 (hit or stand).

    Returns:
        keras.Sequential: The compiled Keras model.
    """
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            # Input layer that matches the shape of the state
            keras.layers.Input(shape=input_shape),

            # First hidden layer with 64 units and ReLU activation
            keras.layers.Dense(64, activation='relu'),

            # Second hidden layer with 64 units and ReLU activation
            keras.layers.Dense(64, activation='relu'),

            # Output layer with a linear activation, one unit per action
            # The output of this layer represents the Q-values
            keras.layers.Dense(num_actions, activation='linear')
        ])
    return model
