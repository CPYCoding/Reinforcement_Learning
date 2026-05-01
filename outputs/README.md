# Assignment 1: Lunar Lander using Deep Q-Learning

This repository contains the implementation and experiment results for training a Deep Q-Network (DQN) agent on the `LunarLander-v3` environment.

## Folder Structure

### `part_a/`
This folder contains the code and results for **Part A**, including the random baseline and the initial DQN implementation.

### `initial result/`
This folder contains the initial/baseline setting used for **Part B and Part C** comparison.

The baseline configuration is:

```python
LEARNING_RATE = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
num_episodes = 2000
```
### `Experiment 1/`

This folder contains the results for Experiment 1: Epsilon Decay.

Different epsilon decay values were tested to observe how the exploration-exploitation balance affects the DQN agent’s performance.

Tested values:

EPSILON_DECAY = 0.990
EPSILON_DECAY = 0.993
EPSILON_DECAY = 0.995
EPSILON_DECAY = 0.997
EPSILON_DECAY = 0.999

### `Experiment 2/`

This folder contains the results for Experiment 2: Learning Rate.

Different learning rates were tested to observe how the update size affects the agent’s learning performance.

Tested values:

LEARNING_RATE = 1e-4
LEARNING_RATE = 5e-4
LEARNING_RATE = 1e-3

### `Experiment 3/`

This folder contains the results for Experiment 3: Target Update Frequency.

Different target update frequencies were tested to observe how often updating the target network affects training stability and final performance.

Tested values:

TARGET_UPDATE_FREQ = 5
TARGET_UPDATE_FREQ = 10
TARGET_UPDATE_FREQ = 20