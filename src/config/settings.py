# blackjack_rl/config/settings.py

# Environment settings
NUM_DECKS = 1  # From original gym.make, sab=True implies single deck dynamics with specific rules
STATE_SIZE = 3
NUM_ACTIONS = 2  # Hit or Stand

# Training parameters
NUM_EPISODES = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 5e-4

# Epsilon-greedy policy parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Replay buffer
REPLAY_BUFFER_CAPACITY = 10000

# Target network update frequency
TARGET_UPDATE_FREQ = 250

# Model saving/loading paths
MODEL_SAVE_PATH = "models/dqn_blackjack.h5"
DOUBLE_DQN_MODEL_SAVE_PATH = "models/double_dqn_blackjack.h5"