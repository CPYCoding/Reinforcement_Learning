import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import csv
from utils import print_stats, plot_baseline, record_episodes

# Hyperparameters (you should experiment with these!)
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997
BATCH_SIZE = 64
BUFFER_SIZE = 50000
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

# baseline_stats = run_random_baseline(num_episodes=100)
# print_stats(baseline_stats)
# plot_baseline(baseline_stats)

# record_episodes(
    num_episodes=5,
    out_dir="outputs/part_a/random_gifs",
    policy_fn=lambda state: env.action_space.sample()
# )

# env.close()
# exit()
# TODO: Implement the classes described in Part B
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=LEARNING_RATE
        )

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.action_dim = action_dim

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state)

        return q_values.argmax().item()
    
    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        current_q = self.q_network(states).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )    

agent = DQNAgent(state_dim, action_dim)

# Training loop
training_log = []
num_episodes = 5000
rewards_history = []
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    episode_losses = []
    
    while not done:
        # TODO: Select action using epsilon-greedy
        action = agent.select_action(state, epsilon)
        # TODO: Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # TODO: Store experience in replay buffer
        agent.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )
        # TODO: Train agent if buffer has enough samples
        loss = agent.train_step()
        if loss is not None:
            episode_losses.append(loss)

        # TODO: Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        state = next_state
        episode_reward += reward

        # pass
    
    # TODO: Decay epsilon
    epsilon = max(
        EPSILON_END,
        epsilon * EPSILON_DECAY
    )
    # TODO: Track and log statistics
    rewards_history.append(episode_reward)
    avg_loss = np.mean(episode_losses) if episode_losses else ""

    training_log.append({
        "episode": episode,
        "reward": episode_reward,
        "epsilon": epsilon,
        "loss": avg_loss
    })
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

os.makedirs("outputs/part_b_c", exist_ok=True)

with open("outputs/part_b_c/training_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["episode", "reward", "epsilon", "loss"]
    )
    writer.writeheader()
    writer.writerows(training_log)

print("Saved training log to outputs/part_b_c/training_log.csv")
torch.save(agent.q_network.state_dict(), "outputs/part_b_c/dqn_lunar_lander.pth")
print("Saved model to outputs/part_b_c/dqn_lunar_lander.pth")

# Testing
# TODO: Test your trained agent
test_rewards = []

for episode in range(100):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, epsilon=0.0)  # No exploration during testing
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    test_rewards.append(total_reward)

print("Test Results:")
print(f"Mean test Reward: {np.mean(test_rewards):.2f}")
print(f"Max Reward: {np.max(test_rewards):.2f}")
print(f"Min Reward: {np.min(test_rewards):.2f}")
print(f"Success Rate: {(np.array(test_rewards) >= 200).mean() * 100:.1f}%")

with open("outputs/part_b_c/test_results.txt", "w") as f:
    f.write("Test Results\n")
    f.write(f"Mean test Reward: {np.mean(test_rewards):.2f}\n")
    f.write(f"Max Reward: {np.max(test_rewards):.2f}\n")
    f.write(f"Min Reward: {np.min(test_rewards):.2f}\n")
    f.write(f"Success Rate: {(np.array(test_rewards) >= 200).mean() * 100:.1f}%\n")

print("Saved test results to outputs/part_b_c/test_results.txt")

# Part C: Plot training curves
episodes = [row["episode"] for row in training_log]
rewards = [row["reward"] for row in training_log]
epsilons = [row["epsilon"] for row in training_log]

losses = [
    row["loss"] if row["loss"] != "" else np.nan
    for row in training_log
]

# Moving average reward
window = 100
moving_avg_rewards = []

for i in range(len(rewards)):
    if i < window:
        moving_avg_rewards.append(np.mean(rewards[:i+1]))
    else:
        moving_avg_rewards.append(np.mean(rewards[i-window+1:i+1]))

os.makedirs("outputs/part_c", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, alpha=0.4, label="Episode Reward")
plt.plot(episodes, moving_avg_rewards, label="100-Episode Moving Average")
plt.axhline(y=200, linestyle="--", label="Solved Threshold (200)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.legend()
plt.grid(True)
plt.savefig("outputs/part_c/reward_curve.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(episodes, losses)
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("outputs/part_c/loss_curve.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(episodes, epsilons)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Curve")
plt.grid(True)
plt.savefig("outputs/part_c/epsilon_curve.png")
plt.close()

print("Saved Part C plots to outputs/part_c/")

env.close()