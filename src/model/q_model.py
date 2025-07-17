# blackjack_rl/model/q_model.py

import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_q_model(input_shape=(3,), num_actions=2):
    """
    Builds and returns a Keras Sequential model for the Q-network.
    """
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_actions)
        ])
    return model