# blackjack_rl/utils/helpers.py

import numpy as np

def smooth(x, w=100):
    """
    Smooths a 1D array using a moving average.
    """
    return np.convolve(x, np.ones(w)/w, mode='valid')