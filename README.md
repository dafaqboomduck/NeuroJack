# NeuroJack

### A Blackjack Reinforcement Learning Project
This project implements a reinforcement learning system to train an agent to play Blackjack. It features a custom environment and two types of Deep Q-Learning (DQN) agents to explore different learning strategies.

## Instalation
You can install the `neurojack` package using pip.
To do this, use the following command in your terminal:
```bash
pip install neurojack
```
**Note**: This package is hosted on the PyPI (Python Package Index) repository. Ensure your pip configuration is up to date and allows installations from https://pypi.org/simple/



---

## 1. `env.py`: The Custom Blackjack Environment

The `BlackjackEnv` class defines a flexible and highly configurable Blackjack game environment. It is built from scratch to allow for specific rule sets and advanced features beyond standard library environments.

### Key Features
- **Customizable Rules**: Supports multiple rule variations, including different numbers of decks, variable payouts, and specific dealer rules.  
- **Card Counting**: Optionally tracks a running count and a true count, enabling more advanced state representations.  
- **State Representation**: The observation space includes:
  - Player’s hand sum  
  - Dealer’s visible card  
  - Usable ace indicator  
  - Optional card counting values (if enabled)  
- **Action Space**: Supports common actions (`Hit`, `Stand`), and optionally `Double Down` or `Split` depending on the rule set.  

---

## 2. `dqn_agent.py`: The Deep Q-Network Agent

The `DQNAgent` implements a Deep Q-Network to learn the optimal strategy for Blackjack.

### How It Works
- **Q-Networks**: Uses two neural networks — a main Q-network for training and a target Q-network for stable updates.  
- **Experience Replay**: Stores past experiences (state, action, reward, next state) in a replay buffer to improve sample efficiency and reduce correlation between updates.  
- **Epsilon-Greedy Policy**: Balances exploration and exploitation by occasionally taking random actions during training.  

---

## 3. `double_dqn_agent.py`: The Double DQN Agent

The `DoubleDQNAgent` extends the standard DQN approach to address overestimation of Q-values, leading to more accurate and stable learning.

### Key Improvements
- **Decoupled Action Selection**: Uses the main Q-network to select actions while the target Q-network evaluates their values.  
- **Stabilized Updates**: This separation reduces bias and improves the accuracy of Q-value estimates.  

---

## Project Structure

- **`NeuroJack.env`** → Custom Blackjack environment  
- **`NeuroJack.dqn_agent`** → Standard DQN agent implementation  
- **`NeuroJack.double_dqn_agent`** → Double DQN agent implementation  

This modular design allows easy switching between different agents and experimentation with various reinforcement learning strategies while keeping the core Blackjack logic consistent.

---

## License

This project is licensed under the [MIT License](LICENSE).  

Copyright (c) 2024 **Razvan Nica** (dafaqboomduck)

You are free to use, modify, and distribute this software in accordance with the terms of the license.  
The software is provided **“as is”**, without warranty of any kind.  

---