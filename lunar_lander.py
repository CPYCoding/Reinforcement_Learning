import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from utils import print_stats, plot_baseline, record_episodes

# Hyperparameters (you should experiment with these!)
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes

# Create environment
env = gym.make('LunarLander-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

def run_random_baseline(num_episodes=100):
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if total_reward >= 200:
            successful_episodes += 1

    stats = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successful_episodes / num_episodes,
    }

    return stats

baseline_stats = run_random_baseline(num_episodes=100)
print_stats(baseline_stats)
plot_baseline(baseline_stats)

record_episodes(
    num_episodes=5,
    out_dir="outputs/part_a/random_gifs",
    policy_fn=lambda state: env.action_space.sample()
)

env.close()
exit()
# TODO: Implement the classes described in Part B


# Training loop
num_episodes = 1000
rewards_history = []
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # TODO: Select action using epsilon-greedy
        # TODO: Take action in environment
        # TODO: Store experience in replay buffer
        # TODO: Train agent if buffer has enough samples
        # TODO: Update target network periodically
        pass
    
    # TODO: Decay epsilon
    # TODO: Track and log statistics
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

# Testing
# TODO: Test your trained agent

env.close()