# blackjack_rl/utils/helpers.py

import numpy as np

def preprocess_state(state):
    """
    Convert state tuple into a normalized float32 vector.
    state = (player_sum: int, dealer_card: int, usable_ace: bool)
    """
    
    player_sum, dealer_card, usable_ace = state
    return np.array([
        (player_sum - 12) / (21 - 12),       # Normalized player sum
        (dealer_card - 1) / (10 - 1),        # Normalized dealer card
        float(usable_ace)                    # Binary feature
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
        raw_state, _ = environment.reset()  # Unpack the first tuple and ignore the dict
        state = preprocess_state(raw_state)
        done = False

        while not done:
            action = agent_instance.choose_action(state)  # Agent acts greedily based on Q-values
            next_state_raw, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
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