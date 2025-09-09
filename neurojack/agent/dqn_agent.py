# blackjack_rl/agent/dqn_agent.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import logging
from tqdm.auto import tqdm 
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt 

# Configure logging for the DQN agent
# The logging level will be set dynamically in __init__ based on the verbose parameter
logger = logging.getLogger(__name__)


# Import required modules
from neurojack.model.q_model import build_q_model
from neurojack.memory.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQNAgent class for training a Deep Q-Network on a Blackjack environment.

    This agent utilizes the principles of Deep Q-Learning to learn an optimal policy
    for playing Blackjack. It uses a main Q-network and a separate target Q-network
    to stabilize the learning process, and a replay buffer to store and sample
    experiences, breaking the correlation between consecutive samples. The agent
    employs an epsilon-greedy strategy for action selection during training to
    balance exploration and exploitation.

    Attributes:
        q_net (keras.Model): The main Q-network that is being trained.
        target_q_net (keras.Model): The target Q-network used for calculating
            the target Q-values, which are used to train the main Q-network.
            This network's weights are periodically updated from the main Q-network.
        replay_buffer (ReplayBuffer): A memory storage for past experiences
            (state, action, reward, next_state, done) from which batches are
            sampled for training.
        optimizer (keras.optimizers.Optimizer): The optimizer used for updating
            the Q-network's weights (e.g., Adam).
        loss_fn (keras.losses.Loss): The loss function used to measure the
            difference between predicted and target Q-values (e.g., Mean Squared Error).
        state_size (int): The dimension of the state representation.
        num_actions (int): The number of possible actions the agent can take.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The current exploration rate for the epsilon-greedy
            policy. It decays over time.
        epsilon_start (float): The initial value of epsilon.
        epsilon_end (float): The minimum value of epsilon.
        epsilon_decay (float): The decay rate for epsilon per episode.
        target_update_freq (int): The frequency (in episodes) at which the
            target Q-network's weights are updated from the main network.
        replay_buffer_capacity (int): The maximum number of experiences the
            replay buffer can hold.
        train_freq (int): The frequency (in global steps) at which the agent
            performs a training step.
        verbose (bool): If True, enables verbose logging and progress bars.
        model_name (str): A descriptive name for the model, used in logging.

    Methods:
        __init__(self, env, learning_rate, gamma, epsilon_start, epsilon_end,
                epsilon_decay, replay_buffer_capacity, target_update_freq,
                train_freq, verbose, model_name, q_net_model):
            Initializes the DQNAgent with environment-specific parameters and
            hyperparameters. It sets up the main and target Q-networks, the
            replay buffer, and the optimizer.

        _preprocess_state(self, state: tuple) -> np.ndarray:
            Converts the raw state from the environment into a normalized
            numerical vector suitable for the Q-network. It handles both
            states with and without card counting.

        choose_action(self, state: np.ndarray, available_actions: list) -> int:
            Selects an action based on an epsilon-greedy policy. With probability
            epsilon, a random valid action is chosen; otherwise, the action with
            the highest predicted Q-value from the Q-network is selected.

        remember(self, state, action, reward, next_state, done):
            Adds a new experience tuple (state, action, reward, next_state, done)
            to the replay buffer.

        _train_step(self, states, actions, rewards, next_states, dones):
            Performs a single gradient descent step to update the main Q-network.
            It calculates the target Q-values using the target network and the Bellman
            equation, then computes the loss between the predicted and target Q-values.
            This method is a TensorFlow graph function for performance.

        update_target_model(self):
            Performs a "hard" update, copying the weights from the main Q-network
            to the target Q-network.

        learn(self, batch_size):
            Samples a batch of experiences from the replay buffer and calls the
            _train_step method to train the Q-network.

        fit(self, env, num_episodes, batch_size, log_interval):
            The main training loop. It runs for a specified number of episodes,
            interacts with the environment, stores experiences, and periodically
            trains the network and updates the target model. It also handles
            epsilon decay and logs training progress.

        evaluate(self, environment, num_eval_episodes, show_win_loss_rates):
            Evaluates the trained agent's performance in a greedy manner
            (epsilon=0) over a number of episodes. It calculates and returns the
            average reward and optionally the win, loss, and push rates.

        _smooth(self, x, w):
            A helper function to apply a moving average for plotting purposes.

        plot_history(self, history, smooth_window):
            Plots the training history of rewards, typically using a smoothed
            version for better visualization.

        save_weights(self, path):
            Saves the weights of the main Q-network to a specified file path.

        load_weights(self, path):
            Loads the weights from a file into both the main and target Q-networks.
    """
    def __init__(self,
                env,  # The environment object
                learning_rate: float = 1e-3,
                gamma: float = 0.9,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.005,
                epsilon_decay: float = 0.99,
                replay_buffer_capacity: int = 10000,
                target_update_freq: int = 100,
                train_freq: int = 1,
                verbose: int = 0,
                model_name: str = "DQN",
                q_net_model: Optional[keras.Model] = None):
        """
        Initializes the DQNAgent with environment-specific parameters and hyperparameters.

        This constructor sets up the core components of the DQN agent, including the
        main and target Q-networks, the replay buffer for experience storage, and the
        optimizer for training. It dynamically determines key parameters from the
        environment to ensure compatibility.

        Args:
            env (BlackjackEnv): The Blackjack environment instance.
            learning_rate (float): The learning rate for the Adam optimizer.
            gamma (float): The discount factor for future rewards.
            epsilon_start (float): The initial value of the exploration rate epsilon.
            epsilon_end (float): The minimum value of epsilon.
            epsilon_decay (float): The multiplicative decay rate for epsilon per episode.
            replay_buffer_capacity (int): The maximum number of experiences to store in the replay buffer.
            target_update_freq (int): The number of episodes after which the target Q-network's weights are updated.
            train_freq (int): The frequency (in global steps) at which a training step is performed.
            verbose (int): If > 0, enables verbose logging and progress bars.
            model_name (str): A descriptive name for the model.
            q_net_model (Optional[keras.Model]): An optional pre-built Keras model for the Q-network.
                                                 If None, a default model is built.
        """

        # Dynamically derive state_size, num_actions, num_decks, and use_card_count from the environment
        self.state_size = env.state_size
        self.num_actions = env.num_actions
        self.num_decks = env.num_decks
        self.use_card_count = env.count_cards

        # Store hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.replay_buffer_capacity = replay_buffer_capacity
        self.train_freq = train_freq
        self.verbose = bool(verbose)
        self.model_name = model_name

        # Set logger level based on verbose
        logger.setLevel(logging.INFO if self.verbose else logging.CRITICAL)

        # Build Q-networks based on the provided model or a default one
        if q_net_model is None:
            # Use the default model if none is provided
            self.q_net = build_q_model(input_shape=(self.state_size,), num_actions=self.num_actions)
        else:
            # Use the custom model provided by the user
            self.q_net = q_net_model
            logger.debug("Using custom Q-network model provided by user.")

        # Create a separate target Q-network with the same architecture and initial weights
        self.target_q_net = keras.models.clone_model(self.q_net)
        self.target_q_net.set_weights(self.q_net.get_weights())

        # Optimizer and Loss
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)

    def _preprocess_state(self, state: tuple) -> np.ndarray:
        """
        Convert a raw state tuple from the environment into a normalized numerical vector.

        This is a crucial step for preparing state observations for a neural network.
        It handles different state structures depending on whether card counting
        is enabled and normalizes the values to a consistent scale (e.g., [0, 1] or [-1, 1]).
        Normalization prevents large values in one feature (e.g., card counts) from
        dominating the network's learning process.

        Args:
            state (tuple): A tuple representing the current state of the game from the environment.
                           Its structure depends on `self.use_card_count`:
                           - If `self.use_card_count` is False: `(player_sum, dealer_card, usable_ace)`
                           - If `self.use_card_count` is True: `(player_sum, dealer_card, usable_ace, running_count, true_count)`

        Returns:
            np.ndarray: A normalized numpy array of type `np.float32` ready for input to the Q-network.
        
        Raises:
            ValueError: If the length of the input state tuple does not match the expected size
                        based on the `self.use_card_count` flag.
        """
        player_sum, dealer_card, usable_ace = 0, 0, 0
        running_count = 0
        true_count = 0

        # Unpack state based on whether card counting is used
        if self.use_card_count:
            if len(state) != 5:
                raise ValueError(f"Expected 5 values in state for card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace, running_count, true_count = state
        else:
            if len(state) != 3:
                raise ValueError(f"Expected 3 values in state without card counting, got {len(state)}: {state}")
            player_sum, dealer_card, usable_ace = state

        # --- Normalize Core Blackjack State Features ---
        # Normalize player_sum (range 2-22) to [0, 1].
        player_sum_norm = (player_sum - 2) / (22 - 2)
        # Normalize dealer_card (range 1-11) to [0, 1].
        dealer_card_norm = (dealer_card - 1) / (11 - 1)
        # Convert usable_ace (boolean) to a float.
        usable_ace_norm = float(usable_ace)

        state_vector = [player_sum_norm, dealer_card_norm, usable_ace_norm]

        # --- Normalize Card Counting Features if Enabled ---
        if self.use_card_count:
            # A conservative upper bound for running count is 10 per deck.
            # Normalizes running count to a range, e.g., [-1, 1].
            max_running_count_abs = self.num_decks * 10
            # Handle potential division by zero if there are no decks, although this is unlikely.
            running_count_norm = running_count / max_running_count_abs if max_running_count_abs > 0 else 0
            
            # The true count is the running count divided by decks remaining.
            # Normalizing it by a fixed value (like 10) ensures it's in a reasonable range for the neural network.
            true_count_norm = true_count / 10.0
            
            state_vector.append(running_count_norm)
            state_vector.append(true_count_norm)

        # Final validation to ensure the processed vector size matches the model's input layer.
        if len(state_vector) != self.state_size:
            raise ValueError(f"Mismatch in processed state vector length ({len(state_vector)}) "
                            f"and agent's expected state_size ({self.state_size}). "
                            "Check DQNAgent initialization and _preprocess_state logic.")

        return np.array(state_vector, dtype=np.float32)

    def choose_action(self, state: np.ndarray, available_actions: list) -> int:
        """
        Chooses an action using an epsilon-greedy policy, respecting available_actions.

        During training, the agent balances exploration and exploitation. With a
        probability of `self.epsilon`, it explores by choosing a random valid action.
        Otherwise, it exploits by selecting the action with the highest Q-value
        predicted by the main Q-network. Actions that are not valid in the current
        game state are masked out.

        Args:
            state (np.ndarray): The preprocessed state vector.
            available_actions (list): A list of booleans indicating which actions are valid.
                                      e.g., `[True, True, False, False]` for Hit, Stand, Double, Split.

        Returns:
            int: The index of the chosen action (e.g., 0 for 'Hit', 1 for 'Stand').
        """
        # Identify the indices of actions that are currently valid
        valid_action_indices = [i for i, valid in enumerate(available_actions) if valid]

        # Handle the case where no valid actions are provided, though this should not happen in normal gameplay
        if not valid_action_indices:
            logger.warning("No valid actions provided. Defaulting to action 0 (Stand).")
            return 0

        # Epsilon-greedy exploration vs. exploitation
        if random.random() < self.epsilon:
            # Exploration: Choose a random valid action
            return random.choice(valid_action_indices)
        else:
            # Exploitation: Choose the best action based on Q-values
            # Reshape the state to match the model's expected input shape (batch_size, state_size)
            state_input = np.expand_dims(state, axis=0)
            # Get Q-values from the Q-network
            q_values = self.q_net(state_input)[0].numpy()

            # Create a masked version of Q-values to ignore invalid actions
            masked_q_values = np.array(q_values)
            for i in range(len(masked_q_values)):
                if i not in valid_action_indices:
                    masked_q_values[i] = -np.inf # Set Q-value of invalid actions to negative infinity

            # Handle edge case where all valid actions have -inf Q-value
            if np.all(masked_q_values == -np.inf):
                logger.warning("All valid actions have -inf Q-value. Falling back to random valid action.")
                return random.choice(valid_action_indices)

            # Return the index of the action with the highest Q-value
            return np.argmax(masked_q_values)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Stores an experience in the replay buffer.

        This method is called after each step in the environment. It takes the
        current state, the action taken, the reward received, the resulting
        next state, and whether the episode is finished, and adds this tuple
        to the agent's replay buffer.

        Args:
            state (np.ndarray): The preprocessed state before the action.
            action (int): The action taken.
            reward (float): The reward received after the action.
            next_state (np.ndarray): The preprocessed state after the action.
            done (bool): A boolean indicating if the episode has ended.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    @tf.function
    def _train_step(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, next_states: tf.Tensor, dones: tf.Tensor) -> tf.Tensor:
        """
        Performs a single training step for the Q-network using the standard DQN update rule.

        This method is decorated with `@tf.function` to compile it into a
        high-performance TensorFlow graph, which significantly speeds up training.
        It computes the target Q-values using the Bellman equation, calculates
        the loss between the predicted and target Q-values, and then applies
        gradient descent to update the main Q-network's weights.

        The Bellman equation for the target is: `target_q = reward + gamma * max(target_q_values(next_state))`
        The loss is the Mean Squared Error between the `predicted_q_values` and the `targets`.

        Args:
            states (tf.Tensor): A batch of states.
            actions (tf.Tensor): A batch of actions taken.
            rewards (tf.Tensor): A batch of rewards received.
            next_states (tf.Tensor): A batch of next states.
            dones (tf.Tensor): A batch of booleans indicating if the episode ended.

        Returns:
            tf.Tensor: The calculated loss for the training step.
        """
        # Ensure correct data types for TensorFlow operations
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        # Calculate target Q-values using the target network
        next_q_values_target = self.target_q_net(next_states)
        max_next_q = tf.reduce_max(next_q_values_target, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            # Get the Q-values for the current states from the main network
            q_values = self.q_net(states)
            # Create indices to gather the Q-values corresponding to the actions taken
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            predicted_q_values = tf.gather_nd(q_values, action_indices)

            # Calculate the loss
            loss = self.loss_fn(targets, predicted_q_values)

        # Compute gradients and update network weights
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def update_target_model(self):
        """
        Updates the target Q-network weights from the main Q-network.

        This is a "hard" update, where the weights of the target Q-network are
        directly copied from the weights of the main Q-network. This process
        stabilizes training by providing a fixed target for a set number of
        episodes, preventing the network from chasing a moving target.
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def learn(self, batch_size: int):
        """
        Samples a batch of experiences from the replay buffer and trains the Q-network.

        This method acts as the entry point for the training process. It first checks
        if the replay buffer contains enough experiences to form a batch. If so, it
        samples a random batch and passes it to the `_train_step` method for
        a single gradient update.

        Args:
            batch_size (int): The number of experiences to sample from the replay buffer.
        """
        # Check if the buffer has enough samples to form a batch
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Perform a single training step
        self._train_step(states, actions, rewards, next_states, dones)

    def fit(self, env, num_episodes: int, batch_size: int, log_interval: int) -> list:
        """
        The main training loop for the DQN agent.

        This function orchestrates the entire training process. It runs for a
        specified number of episodes, interacting with the environment, storing
        experiences in the replay buffer, and periodically training the Q-network
        and updating the target network. It also handles the decay of epsilon
        and logs the training progress.

        Args:
            env (BlackjackEnv): The Blackjack environment to train on.
            num_episodes (int): The total number of episodes to train for.
            batch_size (int): The number of experiences to sample for each training step.
            log_interval (int): The number of episodes after which to log the training progress.

        Returns:
            list: A list containing the total reward for each episode.
        """
        rewards_history = []
        target_update_counter = 0
        steps_in_interval = 0
        current_interval_rewards = []
        batch_num = 0
        global_step_counter = 0 # Counter for total steps to control learn frequency

        logger.info(f"Starting {self.model_name} training for {num_episodes} episodes...")

        pbar_batch = tqdm(total=log_interval, desc=f"Batch {batch_num + 1}/{num_episodes // log_interval}",
                          unit=" episode", leave=True, dynamic_ncols=True, disable=not self.verbose)

        try:
            for episode in range(num_episodes):
                obs, info = env.reset()
                state = self._preprocess_state(obs)
                done = False
                total_reward = 0

                while not done:
                    # Dynamically get available actions from the environment
                    available_actions = [True, True] # 'Hit', 'Stand' are always available
                    if env.allow_doubling:
                        available_actions.append(info.get('can_double', False))
                    if env.allow_splitting:
                        available_actions.append(info.get('can_split', False))

                    # Pad with False if the environment has a smaller number of actions
                    while len(available_actions) < self.num_actions:
                        available_actions.append(False)
                
                    action = self.choose_action(state, available_actions)

                    next_obs, reward, done, info = env.step(action)
                    next_state = self._preprocess_state(next_obs)

                    # Store the experience in the replay buffer
                    self.remember(state, action, reward, next_state, done)

                    # Periodically train the model based on a global step counter
                    global_step_counter += 1
                    if global_step_counter % self.train_freq == 0:
                        self.learn(batch_size)

                    state = next_state
                    total_reward += reward
                    steps_in_interval += 1

                # Decay epsilon at the end of each episode
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

                # Periodically update the target network
                target_update_counter += 1
                if target_update_counter >= self.target_update_freq:
                    self.update_target_model()
                    target_update_counter = 0

                # Log rewards and update progress bar
                rewards_history.append(total_reward)
                current_interval_rewards.append(total_reward)

                pbar_batch.update(1)
                pbar_batch.set_postfix({
                    'AvgR': f"{np.mean(current_interval_rewards):.2f}" if current_interval_rewards else 'N/A',
                    'Eps': f"{self.epsilon:.3f}",
                    'Buf': f"{len(self.replay_buffer)}",
                    'Steps/Int': f"{steps_in_interval}"
                })

                # Log interval-based progress
                if (episode + 1) % log_interval == 0:
                    batch_num += 1
                    avg_reward_interval = np.mean(current_interval_rewards)

                    logger.info(
                        f"\nEpisode Batch {episode + 1}/{num_episodes}, "
                        f"Avg Reward (last {log_interval}): {avg_reward_interval:.4f}"
                    )
                    steps_in_interval = 0
                    current_interval_rewards = []

                    if (episode + 1) < num_episodes:
                        pbar_batch.reset(total=log_interval)
                        pbar_batch.set_description(f"Batch {batch_num + 1}/{num_episodes // log_interval}")
                        pbar_batch.refresh()

        finally:
            pbar_batch.close()

        logger.info(f"{self.model_name} training complete.")
        return rewards_history

    def evaluate(self, environment, num_eval_episodes: int, show_win_loss_rates: bool = False) -> Union[float, Tuple[float, float, float, float]]:
        """
        Evaluates the agent's performance in the given environment using a greedy policy.

        During evaluation, the agent's exploration rate (`epsilon`) is set to 0 to
        ensure it always selects the best-known action. The function runs for a
        specified number of episodes, calculates the average reward, and can
        optionally return and log win, loss, and push rates.

        Args:
            environment: The Blackjack environment to evaluate on.
            num_eval_episodes (int): The number of episodes to run for evaluation.
            show_win_loss_rates (bool): If True, also returns and logs win/loss/push rates.

        Returns:
            Union[float, Tuple[float, float, float, float]]:
                If `show_win_loss_rates` is False, returns the average reward.
                If `show_win_loss_rates` is True, returns a tuple of
                `(average_reward, win_rate, push_rate, loss_rate)`.
        """
        logger.info(f"Starting evaluation for {num_eval_episodes} episodes...")
        total_rewards = []
        wins = 0
        pushes = 0
        losses = 0

        # Save the original epsilon and set to 0 for greedy evaluation
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        with tqdm(range(num_eval_episodes), desc=f"{self.model_name} Evaluation", unit="episode",
                  leave=True, dynamic_ncols=True, disable=not self.verbose) as pbar_eval:
            for episode in pbar_eval:
                raw_state, info = environment.reset()
                state = self._preprocess_state(raw_state)
                done = False
                episode_reward = 0

                while not done:
                    # Determine available actions from the environment
                    available_actions = [True, True]
                    if environment.allow_doubling:
                        available_actions.append(info.get('can_double', False))
                    if environment.allow_splitting:
                        available_actions.append(info.get('can_split', False))
                    while len(available_actions) < self.num_actions:
                        available_actions.append(False)

                    # Choose action using the greedy policy (epsilon=0)
                    action = self.choose_action(state, available_actions)
                    next_state_raw, reward, done, info = environment.step(action)
                    state = self._preprocess_state(next_state_raw)
                    episode_reward += reward

                total_rewards.append(episode_reward)

                # Count wins, pushes, and losses
                if episode_reward > 0:
                    wins += 1
                elif episode_reward < 0:
                    losses += 1
                else:
                    pushes += 1

                # Update evaluation progress bar description
                postfix_dict = {'AvgR': f"{np.mean(total_rewards):.2f}"}
                if show_win_loss_rates:
                    postfix_dict['Wins'] = f"{wins}"
                    postfix_dict['Pushes'] = f"{pushes}"
                    postfix_dict['Losses'] = f"{losses}"
                pbar_eval.set_postfix(postfix_dict)

        # Restore the original epsilon value
        self.epsilon = original_epsilon

        # Calculate final metrics
        total_episodes = len(total_rewards)
        average_reward = np.mean(total_rewards) if total_episodes > 0 else 0.0
        win_rate = wins / total_episodes if total_episodes > 0 else 0
        push_rate = pushes / total_episodes if total_episodes > 0 else 0
        loss_rate = losses / total_episodes if total_episodes > 0 else 0


        logger.info("\n--- Evaluation Results ---")
        logger.info(f"Total Episodes: {total_episodes}")
        logger.info(f"Average Reward: {average_reward:.4f}")
        if show_win_loss_rates:
            logger.info(f"Wins: {wins} ({win_rate:.2%})")
            logger.info(f"Pushes: {pushes} ({push_rate:.2%})")
            logger.info(f"Losses: {losses} ({loss_rate:.2%})")
        logger.info("--------------------------")

        if show_win_loss_rates:
            return average_reward, win_rate, push_rate, loss_rate
        else:
            return average_reward

    def _smooth(self, x, w=100):
        """
        Smooths a 1D array using a moving average.

        This is a helper function used for creating a more readable plot of
        the training rewards, as raw episode-by-episode rewards can be very noisy.
        The `np.convolve` function is used to apply a convolution with a
        sliding window.

        Args:
            x (np.ndarray): The 1D array of data to be smoothed.
            w (int): The size of the moving average window.

        Returns:
            np.ndarray: The smoothed array.
        """
        return np.convolve(x, np.ones(w)/w, mode='valid')

    def plot_history(self, history, smooth_window: int = 1000):
        """
        Plots the training history of rewards.

        This function generates a plot of the average reward over time.
        It uses the `_smooth` helper method to create a clearer visualization
        by applying a moving average to the reward history.

        Args:
            history (list or np.ndarray): A list or array of rewards per episode.
            smooth_window (int): The window size for the moving average smoothing.
        """
        # Plotting the smoothed rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self._smooth(history, w=smooth_window))
        plt.title("DQN Training Rewards (Smoothed)")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.grid(True)
        plt.show()

    def save_weights(self, path: str = None):
        """
        Saves the weights of the main Q-network to a file.

        This method is used to persist the learned policy. The weights are saved
        in a TensorFlow-compatible format to the specified path.

        Args:
            path (str): The file path where the weights will be saved.
                        If `None`, an error is logged.
        """
        if path is None:
            logger.error("Error: A path must be provided to save weights.")
            return
        self.q_net.save_weights(path)
        logger.info(f"{self.model_name} model weights saved to {path}")

    def load_weights(self, path: str = None):
        """
        Loads Q-network weights from a file into both the main and target networks.

        This method is used to load a previously saved policy. After loading
        the weights into the main Q-network, it performs a hard update to
        copy them to the target Q-network, ensuring they are synchronized.

        Args:
            path (str): The file path from which to load the weights.
                        If `None`, an error is logged.
        """
        if path is None:
            logger.error("Error: A path must be provided to load weights.")
            return
        try:
            # Load weights into the main Q-network
            self.q_net.load_weights(path)
            # Synchronize the target network with the main network
            self.target_q_net.set_weights(self.q_net.get_weights())
            logger.info(f"{self.model_name} model weights loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model weights from {path}: {e}")
