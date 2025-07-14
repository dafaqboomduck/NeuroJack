# blackjack_rl/model/q_model.py

import tensorflow as tf
from tensorflow import keras

def build_q_model(input_shape=(3,), num_actions=2):
    """
    Builds and returns a Keras Sequential model for the Q-network.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_actions)
    ])
    return model