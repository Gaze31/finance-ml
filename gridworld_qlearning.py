"""
Q-Learning on a Custom GridWorld
=================================
Author  : ruthless-mentor edition
Python  : 3.9+
Deps    : numpy, matplotlib (stdlib otherwise)

Layout legend
  S  = Start
  G  = Goal  (+1 reward)
  #  = Wall  (impassable)
  .  = Free cell

Actions : 0=UP  1=DOWN  2=LEFT  3=RIGHT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional
import time
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────
# 1. ENVIRONMENT
# ─────────────────────────────────────────────

GRID_DEFAULT = [
    "S . . . # . . . .",
    ". # # . # . # # .",
    ". # . . . . . # .",
    ". . . # # # . . .",
    "# # . # . . . # .",
    ". . . . . # . . .",
    ". # # # . # # . .",
    ". . . . . . . # .",
    ". # . # # # . . G",
]

ACTIONS = {
    0: (-1, 0),  # UP
    1: ( 1, 0),  # DOWN
    2: ( 0,-1),  # LEFT
    3: ( 0, 1),  # RIGHT
}
ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class GridWorld:
    """Deterministic grid environment."""

    def __init__(self, grid_str: list[str]):
        self.grid = [row.split() for row in grid_str]
        self.nrows = len(self.grid)
        self.ncols = len(self.grid[0])
        self.start = self._find("S")
        self.goal  = self._find("G")
        self.walls = {
            (r, c)
            for r in range(self.nrows)
            for c in range(self.ncols)
            if self.grid[r][c] == "#"
        }
        self.n_states  = self.nrows * self.ncols
        self.n_actions = 4
        self._state: tuple[int, int] = self.start

    # ── helpers ──────────────────────────────
    def _find(self, char: str) -> tuple[int, int]:
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == char:
                    return (r, c)
        raise ValueError(f"'{char}' not found in grid")

    def encode(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.ncols + pos[1]

    def decode(self, state: int) -> tuple[int, int]:
        return divmod(state, self.ncols)

    # ── gym-style interface ───────────────────
    def reset(self) -> int:
        self._state = self.start
        return self.encode(self._state)

    def step(self, action: int) -> tuple[int, float, bool]:
        dr, dc = ACTIONS[action]
        nr = self._state[0] + dr
        nc = self._state[1] + dc

        # boundary / wall check → stay in place
        if (
            0 <= nr < self.nrows
            and 0 <= nc < self.ncols
            and (nr, nc) not in self.walls
        ):
            self._state = (nr, nc)

        done   = self._state == self.goal
        reward = 1.0 if done else -0.01   # small step penalty keeps paths short
        return self.encode(self._state), reward, done

    @property
    def state(self) -> int:
        return self.encode(self._state)


# ─────────────────────────────────────────────
# 2. AGENT
# ─────────────────────────────────────────────

@dataclass
class HyperParams:
    alpha:       float = 0.1     # learning rate
    gamma:       float = 0.95    # discount factor
    epsilon:     float = 1.0     # initial exploration
    eps_min:     float = 0.01    # floor
    eps_decay:   float = 0.995   # per-episode multiplicative decay
    n_episodes:  int   = 3_000
    max_steps:   int   = 500


class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration.

    Update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') - Q(s,a) ]
    """

    def __init__(self, n_states: int, n_actions: int, hp: HyperParams):
        self.hp        = hp
        self.n_actions = n_actions
        self.Q         = np.zeros((n_states, n_actions))  # optimistic init = 0
        self.epsilon   = hp.epsilon

    def select_action(self, state: int) -> int:
        """ε-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        target = r if done else r + self.hp.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.hp.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.hp.eps_min, self.epsilon * self.hp.eps_decay)

    @property
    def greedy_policy(self) -> np.ndarray:
        """Returns best action per state."""
        return np.argmax(self.Q, axis=1)


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

@dataclass
class TrainingLog:
    rewards:       list[float] = field(default_factory=list)
    steps:         list[int]   = field(default_factory=list)
    epsilons:      list[float] = field(default_factory=list)
    success_flags: list[bool]  = field(default_factory=list)


def train(
    env: GridWorld,
    agent: QLearningAgent,
    hp: HyperParams,
    verbose: bool = True,
) -> TrainingLog:
    log = TrainingLog()
    t0  = time.perf_counter()

    for ep in range(hp.n_episodes):
        s    = env.reset()
        ep_r = 0.0

        for step in range(hp.max_steps):
            a             = agent.select_action(s)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, s_next, done)
            s    = s_next
            ep_r += r
            if done:
                break

        agent.decay_epsilon()
        log.rewards.append(ep_r)
        log.steps.append(step + 1)
        log.epsilons.append(agent.epsilon)
        log.success_flags.append(done)

        if verbose and (ep + 1) % 500 == 0:
            win_rate = np.mean(log.success_flags[-500:]) * 100
            print(
                f"Episode {ep+1:>5} | "
                f"ε={agent.epsilon:.3f} | "
                f"avg_reward={np.mean(log.rewards[-500:]):+.3f} | "
                f"win_rate={win_rate:.1f}%"
            )

    elapsed = time.perf_counter() - t0
    print(f"\nTraining done in {elapsed:.2f}s")
    return log


# ─────────────────────────────────────────────
# 4. EVALUATION — greedy rollout
# ─────────────────────────────────────────────

def evaluate_greedy(env: GridWorld, agent: QLearningAgent) -> Optional[list[tuple]]:
    """
    Run one greedy episode. Returns path (list of positions) or None if stuck.
    """
    s    = env.reset()
    path = [env.decode(s)]

    for _ in range(env.n_states * 2):           # upper bound = can't loop forever
        a          = int(np.argmax(agent.Q[s]))
        s, _, done = env.step(a)
        path.append(env.decode(s))
        if done:
            return path

    return None   # agent failed


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

def plot_training(log: TrainingLog):
    window = 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Q-Learning Training Metrics", fontsize=14, fontweight="bold")

    # — smoothed reward
    smoothed = np.convolve(log.rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(smoothed, color="#2196F3")
    axes[0].set_title("Smoothed Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    # — steps per episode
    axes[1].plot(log.steps, alpha=0.4, color="#FF5722")
    axes[1].plot(
        np.convolve(log.steps, np.ones(window) / window, mode="valid"),
        color="#FF5722", linewidth=2
    )
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    # — epsilon decay
    axes[2].plot(log.epsilons, color="#4CAF50")
    axes[2].set_title("Epsilon Decay")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("ε")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "training_metrics.png"), dpi=150)
    plt.show()
    print("Saved → training_metrics.png")


def plot_policy(env: GridWorld, agent: QLearningAgent, path: Optional[list]):
    fig, ax = plt.subplots(figsize=(env.ncols * 0.9, env.nrows * 0.9))
    ax.set_xlim(-0.5, env.ncols - 0.5)
    ax.set_ylim(-0.5, env.nrows - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Learned Greedy Policy", fontsize=13, fontweight="bold")

    policy = agent.greedy_policy

    for r in range(env.nrows):
        for c in range(env.ncols):
            cell = (r, c)
            if cell in env.walls:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#37474F"))
            elif cell == env.goal:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#66BB6A"))
                ax.text(c, r, "G", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            elif cell == env.start:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#42A5F5"))
                ax.text(c, r, "S", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            else:
                s      = env.encode(cell)
                action = policy[s]
                ax.text(c, r, ACTION_SYMBOLS[action], ha="center", va="center", fontsize=11, color="#212121")

    # overlay greedy path
    if path:
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, "r-o", markersize=4, linewidth=1.5, alpha=0.7, label="Greedy path")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xticks(range(env.ncols))
    ax.set_yticks(range(env.nrows))
    ax.grid(True, linewidth=0.4, color="#B0BEC5")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "policy.png"), dpi=150)
    plt.show()
    print("Saved → policy.png")


def plot_value_map(env: GridWorld, agent: QLearningAgent):
    V = np.max(agent.Q, axis=1).reshape(env.nrows, env.ncols)
    # mask walls
    mask = np.zeros_like(V)
    for r, c in env.walls:
        mask[r, c] = np.nan
        V[r, c]    = np.nan

    fig, ax = plt.subplots(figsize=(env.ncols * 0.9, env.nrows * 0.9))
    im = ax.imshow(V, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="V(s) = max_a Q(s,a)")
    ax.set_title("State Value Map", fontsize=13, fontweight="bold")
    for r in range(env.nrows):
        for c in range(env.ncols):
            if (r, c) not in env.walls:
                ax.text(c, r, f"{V[r,c]:.2f}", ha="center", va="center", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "value_map.png"), dpi=150)
    plt.show()
    print("Saved → value_map.png")


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    env   = GridWorld(GRID_DEFAULT)
    hp    = HyperParams()
    agent = QLearningAgent(env.n_states, env.n_actions, hp)

    print("=" * 60)
    print(f"  Grid : {env.nrows}×{env.ncols}  |  States: {env.n_states}")
    print(f"  Start: {env.start}  |  Goal: {env.goal}")
    print(f"  Walls: {len(env.walls)}")
    print("=" * 60)

    log  = train(env, agent, hp)
    path = evaluate_greedy(env, agent)

    if path:
        print(f"\nGreedy path found! Length = {len(path)} steps")
    else:
        print("\n⚠ Agent failed to reach goal in greedy rollout — train longer or tune HP.")

    plot_training(log)
    plot_policy(env, agent, path)
    plot_value_map(env, agent)