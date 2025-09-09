# blackjack_rl/memory/replay_buffer.py
"""
This module implements a ReplayBuffer class, a crucial component in off-policy
reinforcement learning algorithms like Deep Q-Networks (DQN). The buffer
stores and allows sampling of past experiences to decorrelate training data
and improve learning stability.
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    A simple replay buffer to store and sample training experiences.

    An experience is a tuple of (state, action, reward, next_state, done).
    """

    def __init__(self, capacity=10000):
        """
        Initializes the ReplayBuffer with a specified capacity.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a single experience to the buffer.

        If the buffer is full, the oldest experience is automatically
        discarded.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The resulting state.
            done (bool): Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing five numpy arrays:
                   (states, actions, rewards, next_states, dones)
        """
        # Ensure the buffer has enough experiences to sample a full batch
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer to sample batch of size %s" % batch_size)

        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch of tuples into separate lists for each element,
        # then convert each list to a NumPy array for efficient processing.
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.

        Returns:
            int: The number of experiences.
        """
        return len(self.buffer)
