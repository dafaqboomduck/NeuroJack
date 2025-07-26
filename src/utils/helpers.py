# blackjack_rl/utils/helpers.py

import random
import numpy as np
import tensorflow as tf
import os

def set_global_seeds(seed):
    """
    Sets seeds for reproducibility across Python, NumPy, and TensorFlow.
    """
    os.environ['PYTHONHASHSEED'] = str(seed) # Set Python hash seed
    random.seed(seed)                        # Python's random module
    np.random.seed(seed)                     # NumPy's random module
    tf.random.set_seed(seed)                 # TensorFlow's random operations
    # For CUDA operations to be deterministic (optional, can impact performance)
    # tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    # tf.config.experimental.enable_op_determinism()
    print(f"Global seeds set to {seed}")

def smooth(x, w=100):
    """s
    Smooths a 1D array using a moving average.
    """
    return np.convolve(x, np.ones(w)/w, mode='valid')