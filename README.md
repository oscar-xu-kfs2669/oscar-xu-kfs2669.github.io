
# Lunar Navigator: Autonomous Landings with Q-Learning and DQN in OpenAI Gym

## Project Overview

This project explores the performance of various Reinforcement Learning (RL) algorithms on the LunarLander-v2 environment from OpenAI Gym. The primary objective is to develop and compare different RL algorithms to solve the LunarLander-v2 problem efficiently and effectively. The algorithms investigated include Q-learning, Monte Carlo, Deep Q-Network (DQN), and DQN with Prioritized Experience Replay (PER).

## Jupyter Notebook and PDF

- **Jupyter Notebook**: [Lunar Navigator - Autonomous Landings with Q-Learning and DQN in OpenAI Gym.ipynb](https://github.com/oscar-xu-kfs2669/Lunar-Navigator/blob/main/Lunar%20Navigator%20-%20Autonomous%20Landings%20with%20Q-Learning%20and%20DQN%20in%20OpenAI%20Gym.ipynb)
- **PDF**: [Lunar Navigator - Autonomous Landings with Q-Learning and DQN in OpenAI Gym.pdf](https://github.com/oscar-xu-kfs2669/Lunar-Navigator/blob/main/Lunar%20Navigator%20-%20Autonomous%20Landings%20with%20Q-Learning%20and%20DQN%20in%20OpenAI%20Gym.pdf)

## Implementation Details

### I. Monte Carlo Method

The Monte Carlo (MC) method for RL is used to estimate the value of each action at a particular state based on many sampled returns following that action from that state.

#### Key Features:
- **Discretization**: Continuous state space is discretized into a finite set of bins.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation.
- **First-Visit Monte Carlo**: Updates the Q-values based on the first occurrence of the state-action pair in an episode.

#### Parameters:
- `gamma = 0.99`
- `epsilon = 1.0`
- `epsilon_min = 0.05`
- `epsilon_decay = 0.999`
- `num_episodes = 10000`

#### Pseudocode:
```python
Initialize Q-table
for each episode in num_episodes:
    state = env.reset()
    episode = []
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    update_q_values(episode, gamma)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

#### Monte Carlo Performance
![Monte Carlo Performance](https://github.com/oscar-xu-kfs2669/oscar-xu-kfs2669.github.io/raw/main/1.Monte-Carlo.gif)

### II. Q-Learning

Q-learning is a model-free RL algorithm that solves decision-making problems by learning an action-value function that gives the expected utility of taking a given action in a given state.

#### Key Features:
- **Q-Table**: Stores Q-values for state-action pairs.
- **Discretization**: Converts continuous state variables into discretized indices.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation.
- **Q-Value Update Rule**: Uses the Bellman equation to update Q-values.

#### Parameters:
- `alpha = 0.5`
- `gamma = 0.99`
- `epsilon = 1.0`
- `epsilon_min = 0.001`
- `epsilon_decay = 0.995`
- `num_episodes = 10000`
- `max_steps_per_episode = 100`

#### Pseudocode:
```python
Initialize Q-table
for each episode in num_episodes:
    state = env.reset()
    for each step in max_steps_per_episode:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        Q[state][action] = Q[state][action] + alpha * (td_target - Q[state][action])
        state = next_state
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

#### Q-Learning Performance
![Q-Learning Performance](https://github.com/oscar-xu-kfs2669/oscar-xu-kfs2669.github.io/raw/main/2.Q-Learning.gif)

### III. Deep Q-Network (DQN)

DQN combines Q-learning with deep neural networks to approximate the Q-value function. It uses a replay buffer to store the agent's experiences and a target network to stabilize training.

#### Key Features:
- **Neural Network**: Approximates Q-values instead of using a Q-table.
- **Replay Buffer**: Stores experiences to break correlation between consecutive samples.
- **Target Network**: Stabilizes training by providing fixed Q-targets.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation.

#### Parameters:
- `gamma = 0.99`
- `epsilon = 1.0`
- `epsilon_min = 0.001`
- `epsilon_decay = 0.995`
- `batch_size = 64`
- `max_steps_per_episode = 500`
- `num_episodes = 1000`
- `update_every = 10`
- `learning_rate = 0.001`

#### Pseudocode:
```python
Initialize replay buffer
Initialize Q-network and target Q-network
for each episode in num_episodes:
    state = env.reset()
    for each step in max_steps_per_episode:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        store_experience(replay_buffer, state, action, reward, next_state, done)
        state = next_state
        if len(replay_buffer) > batch_size:
            batch = sample_experiences(replay_buffer, batch_size)
            train_q_network(batch, gamma)
        if step % update_every == 0:
            update_target_network()
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

#### DQN Performance
![DQN Performance](https://github.com/oscar-xu-kfs2669/oscar-xu-kfs2669.github.io/raw/main/3.DQN.gif)

### IV. DQN with Prioritized Experience Replay (PER)

DQN with PER prioritizes experiences based on their temporal difference (TD) error, allowing the agent to learn more effectively from important experiences.

#### Key Features:
- **Prioritized Replay Buffer**: Prioritizes experiences with higher TD errors.
- **TD Error**: Quantifies the surprise or learning potential of an experience.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation.
- **Neural Network**: Similar architecture to standard DQN.

#### Parameters:
- `gamma = 0.99`
- `epsilon = 1.0`
- `epsilon_min = 0.001`
- `epsilon_decay = 0.995`
- `batch_size = 64`
- `max_steps_per_episode = 500`
- `num_episodes = 1000`
- `update_every = 10`
- `learning_rate = 0.001`
- `alpha = 0.02`

#### Pseudocode:
```python
Initialize prioritized replay buffer
Initialize Q-network and target Q-network
for each episode in num_episodes:
    state = env.reset()
    for each step in max_steps_per_episode:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        td_error = compute_td_error(state, action, reward, next_state, done, gamma)
        store_experience(prioritized_replay_buffer, td_error, state, action, reward, next_state, done)
        state = next_state
        if len(prioritized_replay_buffer) > batch_size:
            batch = sample_experiences(prioritized_replay_buffer, batch_size)
            train_q_network(batch, gamma)
        if step % update_every == 0:
            update_target_network()
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

#### DQN-PER Performance
![DQN-PER Performance](https://github.com/oscar-xu-kfs2669/oscar-xu-kfs2669.github.io/raw/main/4.DQN-PER.gif)

## Results

The performance of the four RL methods was compared based on learning speed, stability, and final performance. DQN and DQN-PER showed significant improvements over traditional methods like Monte Carlo and Q-Learning.

### Comparison of Four RL Methods

![Result](https://github.com/oscar-xu-kfs2669/Lunar-Navigator/blob/main/Average%20Reward%20vs%20Episode.png)

### Analysis
+ Monte Carlo: Shown in yellow, the Monte Carlo method has the lowest performance among the four. It shows an overall trend of increasing reward but remains below zero throughout the episodes, indicating less effective learning in this context.
+ Q-Learning: The pink line represents Q-Learning. Similar to Monte Carlo, its performance fluctuates and generally stays below zero, suggesting limited success in solving the LunarLander-v2 environment effectively.
+ DQN (Deep Q-Network): Represented by the blue line, the DQN method demonstrates significant improvement over Monte Carlo and Q-Learning. It trends upward strongly and surpasses the zero reward threshold early on, maintaining a performance generally above this baseline, which suggests better learning and problem-solving capability.
+ DQN-PER (Deep Q-Network with Prioritized Experience Replay): The purple line indicates the DQN-PER, which shows a pattern similar to the standard DQN but with slightly higher variability. It also trends above the zero reward threshold for much of its trajectory, reflecting effective learning adjustments provided by the prioritized replay mechanism.

## Conclusion

DQN and DQN-PER demonstrated better learning and problem-solving capabilities for the LunarLander-v2 environment. Future work will focus on enhancing computing efficiency and exploring further improvements in RL algorithms.

## Author

- **Oscar Xu** - Master's in Computer Science at Northwestern University
- **GitHub**: [oscar-xu-kfs2669](https://github.com/oscar-xu-kfs2669)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
