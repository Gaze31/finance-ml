"""
DQN Stock Trading Agent
=======================
A complete Deep Q-Network implementation for stock trading using pure NumPy.
No PyTorch/TensorFlow dependency — all forward passes and backprop by hand.

Architecture:
  StockEnv       — RL environment wrapping price data
  ReplayBuffer   — experience replay deque
  DQNNetwork     — 3-layer MLP with ReLU, trained via MSE + SGD
  DQNAgent       — epsilon-greedy policy + target network + training loop
  Backtest       — evaluation and trade logging
"""

import numpy as np
import random
import json
import math
import os
from collections import deque
from typing import List, Tuple, Optional

# Output directory = same folder as this script
_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


# ─── synthetic price data (replace with real OHLCV later) ─────────────────────

def generate_price_series(
    n_days: int = 500,
    start_price: float = 100.0,
    mu: float = 0.0005,
    sigma: float = 0.015,
    seed: int = 42,
) -> np.ndarray:
    """
    Regime-switching price series — alternates bull/bear/sideways phases
    so the agent must learn context, not memorise a fixed seed.
    Includes occasional volatility spikes (earnings-style shocks).
    """
    np.random.seed(seed)
    returns = []
    day = 0
    while day < n_days:
        regime_len = int(np.random.randint(40, 120))
        regime = np.random.choice(["bull", "bear", "sideways"])
        if regime == "bull":
            r_mu    = float(np.random.uniform(0.0005, 0.002))
            r_sigma = float(np.random.uniform(0.008, 0.018))
        elif regime == "bear":
            r_mu    = float(np.random.uniform(-0.002, -0.0003))
            r_sigma = float(np.random.uniform(0.012, 0.025))
        else:
            r_mu    = float(np.random.uniform(-0.0002, 0.0002))
            r_sigma = float(np.random.uniform(0.005, 0.012))
        chunk = np.random.normal(r_mu, r_sigma, regime_len).tolist()
        # Occasional volatility spike
        if np.random.rand() < 0.15:
            spike_idx = np.random.randint(0, regime_len)
            chunk[spike_idx] += float(np.random.choice([-1, 1])) * float(np.random.uniform(0.03, 0.08))
        returns.extend(chunk)
        day += regime_len
    prices = start_price * np.exp(np.cumsum(np.array(returns[:n_days], dtype=np.float32)))
    return prices.astype(np.float32)


# ─── feature engineering ──────────────────────────────────────────────────────

def compute_features(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Build a feature matrix from raw prices.
    Each row = one observation with:
      - normalised price (price / SMA_10 - 1)
      - 5-day return
      - 10-day return
      - 5-day rolling std (volatility proxy)
      - RSI-14
    Returns shape (T, 5) — first (window) rows dropped due to lookback.
    """
    n = len(prices)
    features = []

    def sma(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode='valid')

    def rsi(arr, period=14):
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.convolve(gains, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period) / period, mode='valid')
        rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-9))
        return 100 - 100 / (1 + rs)

    sma10 = sma(prices, window)
    rsi14 = rsi(prices, 14)

    # Align lengths — use the shorter series as the reference
    min_len = min(len(sma10), len(rsi14))

    for i in range(min_len):
        idx = n - min_len + i          # index into original price array
        p = prices[idx]
        s = sma10[len(sma10) - min_len + i]

        price_norm = p / s - 1.0 if s > 0 else 0.0

        ret5  = (prices[idx] / prices[max(0, idx - 5)] - 1.0)  if idx >= 5  else 0.0
        ret10 = (prices[idx] / prices[max(0, idx - 10)] - 1.0) if idx >= 10 else 0.0

        chunk = prices[max(0, idx - 5): idx + 1]
        vol5  = float(np.std(chunk)) / (float(np.mean(chunk)) + 1e-9) if len(chunk) > 1 else 0.0

        rsi_val = rsi14[i] / 100.0 - 0.5   # centre around 0

        features.append([price_norm, ret5, ret10, vol5, rsi_val])

    return np.array(features, dtype=np.float32)


# ─── trading environment ───────────────────────────────────────────────────────

class StockEnv:
    """
    Single-asset discrete trading environment.

    Actions: 0 = HOLD, 1 = BUY (go long), 2 = SELL (close long)

    State: feature vector at current step + [position, unrealised_pnl_pct]
    Reward: realised PnL on sell, daily mark-to-market on open position,
            small penalty for excessive trading.
    """

    HOLD = 0
    BUY  = 1
    SELL = 2

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_cash: float = 10_000.0,
        trade_cost: float = 0.002,      # 0.2% per trade — realistic + anti-churn
        hold_penalty: float = 0.0,
        max_steps: Optional[int] = None,
    ):
        assert len(prices) == len(features), "prices and features must align"
        self.prices        = prices
        self.features      = features
        self.initial_cash  = initial_cash
        self.trade_cost    = trade_cost
        self.hold_penalty  = hold_penalty
        self.max_steps     = max_steps or len(prices) - 1
        self.state_dim     = features.shape[1] + 2   # features + [position, upnl]

        self.reset()

    def reset(self) -> np.ndarray:
        self.step_idx      = 0
        self.cash          = self.initial_cash
        self.position      = 0          # shares held (0 or 1 for simplicity)
        self.entry_price   = 0.0
        self.portfolio_val = self.initial_cash
        self.trade_log: List[dict] = []
        return self._state()

    def _state(self) -> np.ndarray:
        f = self.features[self.step_idx]
        upnl = 0.0
        if self.position == 1:
            upnl = (self.prices[self.step_idx] - self.entry_price) / (self.entry_price + 1e-9)
        return np.concatenate([f, [float(self.position), upnl]], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        price = self.prices[self.step_idx]
        info  = {}

        # ── action masking: invalid actions are silently converted to HOLD ──────
        # This removes the agent's ability to exploit invalid-action edge cases.
        # We log them separately for diagnostics but don't penalise — just ignore.
        if action == self.BUY  and self.position == 1: action = self.HOLD
        if action == self.SELL and self.position == 0: action = self.HOLD

        prev_portfolio = self.cash + (price * self.position)

        tc = 0.0
        if action == self.BUY:
            tc               = price * self.trade_cost
            self.entry_price = price
            self.cash       -= price + tc
            self.position    = 1
            self.trade_log.append({"step": self.step_idx, "action": "BUY",  "price": float(price)})

        elif action == self.SELL:
            tc    = price * self.trade_cost
            pnl   = (price - tc) - self.entry_price
            self.cash       += price - tc
            self.position    = 0
            self.entry_price = 0.0
            self.trade_log.append({"step": self.step_idx, "action": "SELL",
                                   "price": float(price), "pnl": float(pnl)})

        self.step_idx += 1
        done           = self.step_idx >= self.max_steps

        next_price         = self.prices[min(self.step_idx, self.max_steps - 1)]
        self.portfolio_val = self.cash + (next_price * self.position)

        # Reward: log-return of portfolio, clipped to [-0.05, +0.05]
        # Clipping keeps the target range stable without exploding gradients.
        # Log-return is additive and numerically stable across episodes.
        raw_r  = math.log((self.portfolio_val + 1e-9) / (prev_portfolio + 1e-9))
        reward = float(np.clip(raw_r, -0.05, 0.05))

        # Cost nudge: explicit trade cost visible to the agent as reward penalty
        if tc > 0:
            reward -= self.trade_cost

        if done and self.position == 1:
            tc2 = next_price * self.trade_cost
            self.cash        += next_price - tc2
            self.position     = 0
            self.portfolio_val = self.cash

        info["portfolio_value"] = self.portfolio_val
        info["position"]        = self.position
        return self._state(), float(reward), done, info

    def total_return(self) -> float:
        return (self.portfolio_val - self.initial_cash) / self.initial_cash


# ─── replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── neural network (NumPy MLP) ────────────────────────────────────────────────

class DenseLayer:
    """Single fully-connected layer with optional ReLU activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        # He initialisation for ReLU layers
        scale = math.sqrt(2.0 / in_dim)
        self.W  = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.activation = activation
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self._t  = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        if self.activation == "relu":
            self._a = np.maximum(0, self._z)
        else:
            self._a = self._z
        return self._a

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            grad_out = grad_out * (self._z > 0)
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def adam_update(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        for (m, v, g, p) in [
            (self.mW, self.vW, self.dW, 'W'),
            (self.mb, self.vb, self.db, 'b'),
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            if p == 'W':
                self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
            else:
                self.b -= lr * m_hat / (np.sqrt(v_hat) + eps)


class DQNNetwork:
    """
    3-layer MLP: input → 128 → 64 → n_actions
    Trained with MSE loss and Adam optimiser.
    """

    def __init__(self, state_dim: int, n_actions: int = 3, lr: float = 1e-3):
        self.layers = [
            DenseLayer(state_dim, 128, activation="relu"),
            DenseLayer(128, 64,   activation="relu"),
            DenseLayer(64,  n_actions, activation="linear"),
        ]
        self.lr       = lr
        self.n_actions = n_actions

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, state_dim) → (batch, n_actions)"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Single state → Q-values, no gradient tracking needed."""
        x = state.reshape(1, -1)
        return self.forward(x)[0]

    def train_on_batch(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        MSE loss only on the selected action's Q-value.
        Other actions treated as correct (zero gradient contribution).
        """
        q_pred    = self.forward(states)               # (B, n_actions)
        batch_idx = np.arange(len(actions))
        delta     = q_pred[batch_idx, actions] - targets  # TD error

        # Huber loss: quadratic inside (-1,1), linear outside
        # Less sensitive to large occasional errors than pure MSE
        abs_delta = np.abs(delta)
        huber     = np.where(abs_delta <= 1.0, 0.5 * delta**2, abs_delta - 0.5)
        loss      = float(np.mean(huber))

        # Gradient of Huber loss w.r.t. delta
        grad_delta = np.where(abs_delta <= 1.0, delta, np.sign(delta))

        error = np.zeros_like(q_pred)
        error[batch_idx, actions] = grad_delta

        # Backprop
        grad = error / len(actions)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # Global gradient norm clipping (better than per-layer clipping)
        flat = np.concatenate([l.dW.ravel() for l in self.layers] +
                              [l.db.ravel() for l in self.layers])
        gnorm = float(np.linalg.norm(flat))
        if gnorm > 1.0:
            scale = 1.0 / gnorm
            for layer in self.layers:
                layer.dW *= scale
                layer.db *= scale

        for layer in self.layers:
            layer.adam_update(self.lr)

        return loss

    def copy_weights_from(self, other: "DQNNetwork"):
        """Hard copy: θ_target ← θ_online"""
        for dst, src in zip(self.layers, other.layers):
            dst.W[:] = src.W
            dst.b[:] = src.b


# ─── DQN agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN with:
      - Epsilon-greedy exploration (linear decay)
      - Experience replay
      - Target network (hard update every C steps)
    """

    def __init__(
        self,
        state_dim:       int,
        n_actions:       int   = 3,
        lr:              float = 1e-3,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   int   = 500,    # steps to decay over
        batch_size:      int   = 64,
        buffer_capacity: int   = 10_000,
        target_update:   int   = 100,    # hard update every N steps
    ):
        self.gamma          = gamma
        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size
        self.target_update  = target_update
        self.n_actions      = n_actions
        self.steps          = 0
        # Mirror StockEnv action constants for masking
        self.HOLD = 0; self.BUY = 1; self.SELL = 2

        self.online  = DQNNetwork(state_dim, n_actions, lr)
        self.target  = DQNNetwork(state_dim, n_actions, lr)
        self.target.copy_weights_from(self.online)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.losses: List[float] = []

    def select_action(self, state: np.ndarray, position: int = -1) -> int:
        """
        Epsilon-greedy with action masking.
        Pass position (0=flat, 1=long) to exclude structurally invalid actions.
        If position unknown (-1), all actions are valid.
        """
        # Build mask: True = action is valid
        mask = np.ones(self.n_actions, dtype=bool)
        if position == 0:
            mask[self.SELL] = False   # can't sell when flat
        elif position == 1:
            mask[self.BUY]  = False   # can't buy when already long

        valid_actions = np.where(mask)[0]

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        q = self.online.predict(state)
        q_masked = np.where(mask, q, -np.inf)
        return int(np.argmax(q_masked))

    def _decay_epsilon(self):
        # Pure linear decay: 1.0 → epsilon_end over exactly epsilon_decay steps
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (1.0 - self.epsilon_end) * self.steps / self.epsilon_decay,
        )

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Double DQN: online selects action, target evaluates value
        next_q_online = self.online.forward(next_states)        # (B, A)
        best_actions  = np.argmax(next_q_online, axis=1)        # (B,)
        next_q_target = self.target.forward(next_states)        # (B, A)
        best_next_q   = next_q_target[np.arange(self.batch_size), best_actions]

        targets = rewards + self.gamma * best_next_q * (1 - dones)

        loss = self.online.train_on_batch(states, targets, actions)
        self.losses.append(loss)

        self.steps += 1
        self._decay_epsilon()

        if self.steps % self.target_update == 0:
            self.target.copy_weights_from(self.online)

        return loss


# ─── training loop ─────────────────────────────────────────────────────────────

def warm_start(agent: DQNAgent, env: StockEnv, n_steps: int = 2000):
    """
    Fill the replay buffer with random experience before training begins.
    Guarantees the agent has seen both BUY and SELL rewards before ε decays.
    """
    state = env.reset()
    for _ in range(n_steps):
        # Force balanced random actions respecting position
        if env.position == 0:
            action = np.random.choice([StockEnv.HOLD, StockEnv.BUY])
        else:
            action = np.random.choice([StockEnv.HOLD, StockEnv.SELL])
        next_state, reward, done, info = env.step(action)
        agent.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()


def train(
    agent:      DQNAgent,
    env:        StockEnv,
    n_episodes: int = 50,
    verbose:    bool = True,
) -> List[dict]:
    # Fill buffer with balanced random experience before epsilon starts decaying
    warm_start(agent, env, n_steps=min(5000, len(env.prices) * 5))

    history = []

    for ep in range(n_episodes):
        state = env.reset()
        done  = False

        while not done:
            action     = agent.select_action(state, position=env.position)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward * 100.0, next_state, done)
            agent.train_step()
            state = next_state

        ret      = env.total_return()
        n_trades = len(env.trade_log)
        avg_loss = float(np.mean(agent.losses[-100:])) if agent.losses else 0.0

        record = {
            "episode":    ep + 1,
            "return_pct": round(ret * 100, 2),
            "n_trades":   n_trades,
            "epsilon":    round(agent.epsilon, 4),
            "loss":       round(avg_loss, 6),
        }
        history.append(record)

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"Ep {ep+1:3d} | return {ret*100:+6.2f}% | "
                f"trades {n_trades:3d} | ε {agent.epsilon:.3f} | loss {avg_loss:.6f}"
            )

    return history


# ─── backtest (greedy policy, no exploration) ──────────────────────────────────

def backtest(
    agent:    DQNAgent,
    env:      StockEnv,
    verbose:  bool = True,
) -> dict:
    state    = env.reset()
    done     = False
    saved_eps = agent.epsilon
    agent.epsilon = 0.0       # pure greedy

    values   = [env.initial_cash]
    actions_taken = []

    while not done:
        action = agent.select_action(state, position=env.position)
        state, _, done, info = env.step(action)
        values.append(info["portfolio_value"])
        actions_taken.append(action)

    agent.epsilon = saved_eps

    ret     = env.total_return()
    n_buys  = actions_taken.count(StockEnv.BUY)
    n_sells = actions_taken.count(StockEnv.SELL)

    # Buy-and-hold benchmark
    bnh_return = (env.prices[-1] - env.prices[0]) / env.prices[0]

    # Sharpe (approximate, daily)
    port_arr = np.array(values, dtype=np.float32)
    daily_r  = np.diff(port_arr) / (port_arr[:-1] + 1e-9)
    sharpe   = float(np.mean(daily_r) / (np.std(daily_r) + 1e-9) * math.sqrt(252))

    # Max drawdown
    peak = np.maximum.accumulate(port_arr)
    dd   = (port_arr - peak) / (peak + 1e-9)
    max_dd = float(np.min(dd))

    result = {
        "final_portfolio": round(float(values[-1]), 2),
        "total_return_pct": round(ret * 100, 2),
        "bnh_return_pct":   round(bnh_return * 100, 2),
        "alpha_pct":        round((ret - bnh_return) * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_buys":           n_buys,
        "n_sells":          n_sells,
        "trade_log":        env.trade_log,
        "portfolio_values": values,
    }

    if verbose:
        print("\n─── Backtest Results ───────────────────────────────")
        print(f"  Final portfolio : ${result['final_portfolio']:,.2f}")
        print(f"  Total return    : {result['total_return_pct']:+.2f}%")
        print(f"  B&H return      : {result['bnh_return_pct']:+.2f}%")
        print(f"  Alpha           : {result['alpha_pct']:+.2f}%")
        print(f"  Sharpe ratio    : {result['sharpe_ratio']:.3f}")
        print(f"  Max drawdown    : {result['max_drawdown_pct']:.2f}%")
        print(f"  Trades (B/S)    : {n_buys} / {n_sells}")
        print("────────────────────────────────────────────────────")

    return result




# ─── results plot ──────────────────────────────────────────────────────────────

def plot_results(history: List[dict], result: dict, out_dir: str) -> None:
    """
    Saves two PNG files to out_dir:
      training_curve.png  — episode return, trade count, loss over training
      equity_curve.png    — backtest portfolio vs buy-and-hold
    Falls back gracefully if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    # ── training curve ──────────────────────────────────────────────────────
    eps     = [r["episode"]    for r in history]
    returns = [r["return_pct"] for r in history]
    trades  = [r["n_trades"]   for r in history]
    losses  = [r["loss"]       for r in history]

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("DQN Stock Trader — Training", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # Return
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eps, returns, color="#2563EB", linewidth=1.2, alpha=0.8)
    ax1.axhline(0, color="#94A3B8", linewidth=0.6, linestyle="--")
    # 10-ep rolling mean
    if len(returns) >= 10:
        rm = np.convolve(returns, np.ones(10)/10, mode="valid")
        ax1.plot(eps[9:], rm, color="#DC2626", linewidth=1.8, label="10-ep avg")
        ax1.legend(fontsize=9)
    ax1.set_ylabel("Return %", fontsize=9)
    ax1.set_title("Episode return", fontsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # Trade count
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(eps, trades, color="#7C3AED", alpha=0.6, width=0.8)
    ax2.set_ylabel("# Trades", fontsize=9)
    ax2.set_title("Trades per episode", fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    # Loss
    ax3 = fig.add_subplot(gs[2])
    ax3.semilogy(eps, [max(l, 1e-9) for l in losses], color="#059669", linewidth=1.2)
    ax3.set_ylabel("Loss (log)", fontsize=9)
    ax3.set_xlabel("Episode", fontsize=9)
    ax3.set_title("Huber loss", fontsize=10)
    ax3.grid(True, alpha=0.3, linewidth=0.5)

    path1 = os.path.join(out_dir, "training_curve.png")
    fig.savefig(path1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── equity curve ────────────────────────────────────────────────────────
    pv        = result["portfolio_values"]
    n         = len(pv)
    init      = pv[0]
    bnh_start = init
    bnh_end   = init * (1 + result["bnh_return_pct"] / 100)
    bnh       = np.linspace(bnh_start, bnh_end, n)

    fig2, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n), pv,  color="#2563EB", linewidth=1.5, label="DQN agent")
    ax.plot(range(n), bnh, color="#94A3B8", linewidth=1.2, linestyle="--", label="Buy & hold")
    ax.axhline(init, color="#E5E7EB", linewidth=0.6)

    # Mark trades
    tlog = result.get("trade_log", [])
    for t in tlog:
        x   = t["step"]
        col = "#16A34A" if t["action"] == "BUY" else "#DC2626"
        mk  = "^" if t["action"] == "BUY" else "v"
        if x < n:
            ax.scatter(x, pv[x], color=col, marker=mk, s=60, zorder=5)

    ax.set_title(
        f"Backtest equity curve   |   Agent {result['total_return_pct']:+.2f}%   "
        f"B&H {result['bnh_return_pct']:+.2f}%   Alpha {result['alpha_pct']:+.2f}%",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Portfolio value ($)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    path2 = os.path.join(out_dir, "equity_curve.png")
    fig2.savefig(path2, dpi=140, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: training_curve.png  equity_curve.png")

# ─── entry point ───────────────────────────────────────────────────────────────

def main():
    set_seed(42)

    print("DQN Stock Trader — NumPy implementation")
    print("=" * 50)

    # 1. Regime-switching price series — 2000 days gives enough bull/bear cycles
    prices   = generate_price_series(n_days=2000, seed=42)
    features = compute_features(prices, window=10)
    prices   = prices[-len(features):]

    # 2. Train / test split (80 / 20)
    split       = int(len(prices) * 0.8)
    tr_prices   = prices[:split];  tr_features = features[:split]
    te_prices   = prices[split:];  te_features = features[split:]

    print(f"Train steps: {split}  |  Test steps: {len(te_prices)}")
    print(f"State dim  : {tr_features.shape[1] + 2}")

    train_env = StockEnv(tr_prices, tr_features, initial_cash=10_000)
    test_env  = StockEnv(te_prices, te_features, initial_cash=10_000)

    # Decay over first 25% of training — fast convergence to exploitation phase
    # Buffer stays small so recent (post-exploration) experience dominates
    total_steps   = split * 200
    epsilon_decay = int(total_steps * 0.25)

    agent = DQNAgent(
        state_dim       = train_env.state_dim,
        n_actions       = 3,
        lr              = 3e-4,
        gamma           = 0.99,
        epsilon_start   = 1.0,
        epsilon_end     = 0.05,
        epsilon_decay   = epsilon_decay,
        batch_size      = 64,
        buffer_capacity = 15_000,   # ~15 full episodes — stays fresh
        target_update   = 100,
    )

    print(f"Epsilon decay over {epsilon_decay:,} steps (~ep {epsilon_decay // split:.0f})\n")

    # 4. Train
    print("Training...\n")
    history = train(agent, train_env, n_episodes=200, verbose=True)

    # 5. Backtest on unseen data
    result = backtest(agent, test_env, verbose=True)

    # 6. Save artefacts
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)):     return int(obj)
            return super().default(obj)

    with open(os.path.join(_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    with open(os.path.join(_DIR, "backtest_result.json"), "w") as f:
        out = {k: v for k, v in result.items() if k != "portfolio_values"}
        json.dump(out, f, indent=2, cls=NumpyEncoder)

    print("\nSaved: training_history.json  backtest_result.json")

    # 7. Plot
    plot_results(history, result, _DIR)

    return history, result


if __name__ == "__main__":
    history, result = main()