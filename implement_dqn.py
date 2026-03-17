import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import  gymnasium as gym

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class DuelingDQN(nn.Module):
    """Dueling DQN Architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        """Forward pass with dueling architecture"""
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """DQN Agent with training capabilities"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        use_dueling: bool = False,
        device: str = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        if use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dims[0]).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dims[0]).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.losses = []
        self.episode_rewards = []
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """Update network using replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(
        self,
        env,
        num_episodes: int = 500,
        max_steps: int = 500,
        verbose: bool = True
    ):
        """Train the agent"""
        for episode in range(num_episodes):
            state, _ = env.reset() if hasattr(env.reset(), '__iter__') and len(env.reset()) == 2 else (env.reset(), None)
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                result = env.step(action)
                
                # Handle different gym versions
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Update network
                loss = self.update()
                if loss > 0:
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store stats
            self.episode_rewards.append(episode_reward)
            if episode_losses:
                self.losses.append(np.mean(episode_losses))
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """Evaluate the agent"""
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset() if hasattr(env.reset(), '__iter__') and len(env.reset()) == 2 else (env.reset(), None)
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                result = env.step(action)
                
                if len(result) == 5:
                    state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = result
                
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def plot_training(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot rewards
        axes[0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        
        # Plot moving average
        window = 20
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(self.episode_rewards)), 
                        moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot losses
        if self.losses:
            axes[1].plot(self.losses, alpha=0.6, color='orange')
            axes[1].set_xlabel('Update Step')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


def demo_cartpole():
    """Demo: Train DQN on CartPole"""
    print("=" * 70)
    print("DQN TRAINING ON CARTPOLE-V1")
    print("=" * 70)
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    )
    
    print(f"\nAgent created on device: {agent.device}")
    print("\nStarting training...")
    
    # Train
    agent.train(env, num_episodes=300, max_steps=500, verbose=True)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    avg_reward = agent.evaluate(env, num_episodes=10)
    print(f"\nAverage reward over 10 episodes: {avg_reward:.2f}")
    
    # Plot results
    agent.plot_training()
    
    env.close()


def demo_mountain_car():
    """Demo: Train DQN on MountainCar"""
    print("=" * 70)
    print("DQN TRAINING ON MOUNTAINCAR-V0")
    print("=" * 70)
    
    # Create environment
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\nEnvironment: MountainCar-v0")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent with Dueling DQN
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=20000,
        batch_size=64,
        target_update=10,
        use_dueling=True
    )
    
    print(f"\nAgent created on device: {agent.device}")
    print("Using Dueling DQN architecture")
    print("\nStarting training...")
    
    # Train
    agent.train(env, num_episodes=500, max_steps=200, verbose=True)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    avg_reward = agent.evaluate(env, num_episodes=10)
    print(f"\nAverage reward over 10 episodes: {avg_reward:.2f}")
    
    # Plot results
    agent.plot_training()
    
    env.close()


if __name__ == "__main__":
    # Run CartPole demo
    demo_cartpole()
    
    # Uncomment to run MountainCar demo
    # demo_mountain_car()