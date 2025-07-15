# blackjack_rl/config/settings.py

# Environment settings
NUM_DECKS = 6
STATE_SIZE = 4  # (player_sum, dealer_card_showing, usable_ace, running_count)
NUM_ACTIONS = 4 # (Stand, Hit, Double Down, Split)

# Training parameters
NUM_EPISODES = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4

# Epsilon-greedy policy parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Replay buffer
REPLAY_BUFFER_CAPACITY = 10000

# Target network update frequency
TARGET_UPDATE_FREQ = 250