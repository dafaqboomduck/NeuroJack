# blackjack_rl/utils/helpers.py

import numpy as np

def preprocess_state(state):
    """
    Convert state tuple into a normalized float32 vector.
    state = (player_sum: int, dealer_card: int, usable_ace: bool, running_count: int)
    
    NOTE: This preprocessing function needs to be updated to handle the new
    running_count feature in the observation space.
    """
    
    player_sum, dealer_card, usable_ace, running_count = state
    # Ensure normalization is appropriate for the new running_count range
    # Assuming running_count can range from -120 to 120 for 6 decks (approx)
    # You might need to adjust the normalization for running_count based on your num_decks
    max_running_count_abs = 20 * 6 # Max absolute count for 6 decks (approx)
    
    return np.array([
        (player_sum - 12) / (21 - 12),       # Normalized player sum (assuming min 12, max 21 relevant)
        (dealer_card - 1) / (10 - 1),        # Normalized dealer card (1-10, Ace is 11, so 1-11 range is better)
        float(usable_ace),                   # Binary feature
        running_count / max_running_count_abs # Normalized running count
    ], dtype=np.float32)

def smooth(x, w=100):
    """
    Smooths a 1D array using a moving average.
    """
    return np.convolve(x, np.ones(w)/w, mode='valid')

# The evaluate_agent logic
def evaluate_agent_in_notebook(agent_instance, environment, num_eval_episodes):
    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    wins = 0
    pushes = 0
    losses = 0

    # Temporarily set epsilon to 0 for evaluation (no exploration)
    original_epsilon = agent_instance.epsilon
    agent_instance.epsilon = 0.0

    for episode in range(num_eval_episodes):
        # Reset now returns (raw_state, info)
        raw_state, _ = environment.reset()  # Unpack the first tuple and ignore the dict
        state = preprocess_state(raw_state)
        done = False

        while not done:
            action = agent_instance.choose_action(state)  # Agent acts greedily based on Q-values
            # Removed 'terminated' and 'truncated' from the unpacking
            next_state_raw, reward, done, _ = environment.step(action) # Removed 'terminated' and 'truncated'
            state = preprocess_state(next_state_raw)

        if reward == 1:
            wins += 1
        elif reward == 0:
            pushes += 1
        else: # reward == -1
            losses += 1

        if (episode + 1) % 1000 == 0:
            print(f"  Evaluation episode {episode + 1}/{num_eval_episodes}")

    agent_instance.epsilon = original_epsilon # Restore original epsilon

    total_episodes = wins + pushes + losses
    win_rate = wins / total_episodes
    push_rate = pushes / total_episodes
    loss_rate = losses / total_episodes

    print("\n--- Evaluation Results ---")
    print(f"Total Episodes: {total_episodes}")
    print(f"Wins: {wins} ({win_rate:.2%})")
    print(f"Pushes: {pushes} ({push_rate:.2%})")
    print(f"Losses: {losses} ({loss_rate:.2%})")
    print("--------------------------")
    return win_rate, push_rate, loss_rate