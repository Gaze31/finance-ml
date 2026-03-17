import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

class QLearningAgent:
    """Q-Learning Agent implementation"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilons = []
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps=1000):
        """Train agent for one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = self.get_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            self.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        self.decay_epsilon()
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilons.append(self.epsilon)
        
        return total_reward, steps
    
    def test_episode(self, env, max_steps=1000, render=False):
        """Test agent (no exploration)"""
        state = env.reset()
        total_reward = 0
        steps = 0
        trajectory = [state]
        
        for step in range(max_steps):
            action = self.get_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            if render:
                env.render()
            
            total_reward += reward
            steps += 1
            state = next_state
            trajectory.append(state)
            
            if done:
                break
        
        return total_reward, steps, trajectory


class GridWorld:
    """Grid World Environment for Q-Learning"""
    
    def __init__(self, size=5, obstacles=None, goal=None):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Down, Left, Right
        
        # Set obstacles and goal
        self.obstacles = obstacles if obstacles else []
        self.goal = goal if goal else (size-1, size-1)
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment to starting position"""
        self.position = (0, 0)
        return self._get_state()
    
    def _get_state(self):
        """Convert position to state index"""
        return self.position[0] * self.size + self.position[1]
    
    def _get_position(self, state):
        """Convert state index to position"""
        return (state // self.size, state % self.size)
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        row, col = self.position
        d_row, d_col = self.actions[action]
        
        new_row = max(0, min(self.size - 1, row + d_row))
        new_col = max(0, min(self.size - 1, col + d_col))
        new_position = (new_row, new_col)
        
        # Check if new position is valid
        if new_position in self.obstacles:
            new_position = self.position  # Stay in place
            reward = -1  # Penalty for hitting obstacle
        elif new_position == self.goal:
            reward = 100  # Large reward for reaching goal
        else:
            reward = -0.1  # Small penalty for each step
        
        self.position = new_position
        done = (self.position == self.goal)
        
        return self._get_state(), reward, done
    
    def render(self):
        """Visualize current state"""
        grid = np.zeros((self.size, self.size))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goal
        grid[self.goal] = 2
        
        # Mark agent position
        grid[self.position] = 1
        
        print("\n" + "="*30)
        for row in grid:
            print(" ".join([
                "🟦" if x == 1 else  # Agent
                "🎯" if x == 2 else  # Goal
                "⬛" if x == -1 else # Obstacle
                "⬜"                 # Empty
                for x in row
            ]))
        print("="*30)


class FrozenLake:
    """Frozen Lake Environment (simplified)"""
    
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Down, Left, Right
        
        # Define holes and goal
        self.holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
        self.goal = (3, 3)
        
        # Action mappings (with slippery ice simulation)
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        self.reset()
    
    def reset(self):
        """Reset to start position"""
        self.position = (0, 0)
        return self._get_state()
    
    def _get_state(self):
        """Convert position to state"""
        return self.position[0] * self.size + self.position[1]
    
    def step(self, action):
        """Take action (with 80% probability, 10% each perpendicular)"""
        # Slippery ice: 20% chance of moving perpendicular
        if np.random.random() < 0.8:
            actual_action = action
        else:
            # Move perpendicular
            if action in [0, 1]:  # Up/Down
                actual_action = np.random.choice([2, 3])  # Left/Right
            else:  # Left/Right
                actual_action = np.random.choice([0, 1])  # Up/Down
        
        row, col = self.position
        d_row, d_col = self.actions[actual_action]
        
        new_row = max(0, min(self.size - 1, row + d_row))
        new_col = max(0, min(self.size - 1, col + d_col))
        new_position = (new_row, new_col)
        
        self.position = new_position
        
        # Determine reward and done
        if self.position in self.holes:
            reward = -10
            done = True
        elif self.position == self.goal:
            reward = 100
            done = True
        else:
            reward = -0.1
            done = False
        
        return self._get_state(), reward, done


def train_agent(env, agent, n_episodes=1000, print_every=100):
    """Train Q-Learning agent"""
    print(f"\n{'='*60}")
    print(f"Training Q-Learning Agent for {n_episodes} episodes")
    print(f"{'='*60}\n")
    
    for episode in range(n_episodes):
        reward, steps = agent.train_episode(env)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(agent.episode_rewards[-print_every:])
            avg_steps = np.mean(agent.episode_lengths[-print_every:])
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Steps: {avg_steps:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


def visualize_training(agent):
    """Visualize training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning Training Progress', fontsize=16, fontweight='bold')
    
    # Moving average window
    window = 50
    
    # Episode rewards
    ax1 = axes[0, 0]
    if len(agent.episode_rewards) > window:
        moving_avg = np.convolve(agent.episode_rewards, 
                                np.ones(window)/window, mode='valid')
        ax1.plot(agent.episode_rewards, alpha=0.3, color='blue', label='Raw')
        ax1.plot(range(window-1, len(agent.episode_rewards)), 
                moving_avg, color='red', linewidth=2, label=f'{window}-Episode MA')
    else:
        ax1.plot(agent.episode_rewards, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ax2 = axes[0, 1]
    if len(agent.episode_lengths) > window:
        moving_avg = np.convolve(agent.episode_lengths, 
                                np.ones(window)/window, mode='valid')
        ax2.plot(agent.episode_lengths, alpha=0.3, color='green', label='Raw')
        ax2.plot(range(window-1, len(agent.episode_lengths)), 
                moving_avg, color='red', linewidth=2, label=f'{window}-Episode MA')
    else:
        ax2.plot(agent.episode_lengths, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax3 = axes[1, 0]
    ax3.plot(agent.epsilons, color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon) Decay')
    ax3.grid(True, alpha=0.3)
    
    # Success rate (last 100 episodes)
    ax4 = axes[1, 1]
    success_window = 100
    if len(agent.episode_rewards) >= success_window:
        success_rates = []
        for i in range(success_window, len(agent.episode_rewards) + 1):
            recent_rewards = agent.episode_rewards[i-success_window:i]
            success_rate = sum(r > 0 for r in recent_rewards) / success_window * 100
            success_rates.append(success_rate)
        
        ax4.plot(range(success_window, len(agent.episode_rewards) + 1), 
                success_rates, color='orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title(f'Success Rate (Last {success_window} Episodes)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_q_table(agent, env, title="Q-Table Heatmap"):
    """Visualize Q-table as heatmap"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Q-values heatmap
    ax1 = axes[0]
    sns.heatmap(agent.q_table, annot=False, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax1, cbar_kws={'label': 'Q-Value'})
    ax1.set_xlabel('Action')
    ax1.set_ylabel('State')
    ax1.set_title(f'{title} - All Q-Values')
    ax1.set_xticklabels(['Up', 'Down', 'Left', 'Right'])
    
    # Best action per state
    ax2 = axes[1]
    best_actions = np.argmax(agent.q_table, axis=1)
    
    if isinstance(env, GridWorld):
        size = env.size
        best_action_grid = best_actions.reshape((size, size))
        
        # Create visualization
        grid_visual = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if (i, j) == env.goal:
                    grid_visual[i, j] = 5  # Goal
                elif (i, j) in env.obstacles:
                    grid_visual[i, j] = -1  # Obstacle
                else:
                    grid_visual[i, j] = best_action_grid[i, j]
        
        im = ax2.imshow(grid_visual, cmap='viridis')
        
        # Add arrows for best actions
        arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        for i in range(size):
            for j in range(size):
                if (i, j) == env.goal:
                    ax2.text(j, i, '🎯', ha='center', va='center', fontsize=20)
                elif (i, j) in env.obstacles:
                    ax2.text(j, i, '⬛', ha='center', va='center', fontsize=20)
                else:
                    action = best_action_grid[i, j]
                    ax2.text(j, i, arrow_map[action], ha='center', va='center',
                            fontsize=20, fontweight='bold', color='white')
        
        ax2.set_title('Policy Visualization (Best Actions)')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def demonstrate_gridworld():
    """Demonstrate Q-Learning on GridWorld"""
    print("\n" + "="*60)
    print("DEMO 1: GRID WORLD ENVIRONMENT")
    print("="*60)
    
    # Create environment
    obstacles = [(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)]
    env = GridWorld(size=5, obstacles=obstacles, goal=(4, 4))
    
    # Create agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agent
    train_agent(env, agent, n_episodes=500, print_every=100)
    
    # Visualize results
    visualize_training(agent)
    visualize_q_table(agent, env, title="GridWorld Q-Table")
    
    # Test trained agent
    print("\nTesting trained agent (5 episodes):")
    for i in range(5):
        reward, steps, trajectory = agent.test_episode(env, render=False)
        print(f"Test Episode {i+1}: Reward = {reward:.1f}, Steps = {steps}")


def demonstrate_frozen_lake():
    """Demonstrate Q-Learning on Frozen Lake"""
    print("\n" + "="*60)
    print("DEMO 2: FROZEN LAKE ENVIRONMENT")
    print("="*60)
    
    # Create environment
    env = FrozenLake(size=4)
    
    # Create agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.01
    )
    
    # Train agent
    train_agent(env, agent, n_episodes=2000, print_every=400)
    
    # Visualize results
    visualize_training(agent)
    
    # Test trained agent
    print("\nTesting trained agent (10 episodes):")
    successes = 0
    for i in range(10):
        reward, steps, trajectory = agent.test_episode(env, render=False)
        success = reward > 0
        successes += success
        print(f"Test Episode {i+1}: {'SUCCESS' if success else 'FAILED'} "
              f"(Reward = {reward:.1f}, Steps = {steps})")
    
    print(f"\nSuccess Rate: {successes}/10 ({successes*10}%)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Q-LEARNING REINFORCEMENT LEARNING DEMONSTRATIONS")
    print("="*60)
    
    # Run demonstrations
    demonstrate_gridworld()
    print("\n" + "="*60 + "\n")
    demonstrate_frozen_lake()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("="*60)