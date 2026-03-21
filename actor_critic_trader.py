"""
Actor-Critic (A2C) Stock Trading Agent
=======================================
Pure NumPy implementation — no PyTorch / TensorFlow required.

Architecture
------------
  ActorNetwork   — policy π(a|s; θ_π): softmax over {HOLD, BUY, SELL}
  CriticNetwork  — value function V(s; θ_v): scalar baseline
  A2CAgent       — shared rollout buffer, GAE advantage, entropy bonus
  StockEnv       — same regime-switching environment as the DQN version

Key differences from DQN
-------------------------
  - No replay buffer. Learns on-policy from rollout trajectories.
  - Separate loss functions: actor uses policy gradient, critic uses MSE.
  - Advantage A(s,a) = G_t - V(s) reduces variance vs raw returns.
  - GAE (Generalised Advantage Estimation) trades bias for further variance reduction.
  - Entropy bonus H(π) keeps the policy from collapsing to a single action.
  - No target network needed — on-policy learning is inherently stable.
"""

import numpy as np
import random
import math
import json
import os
from typing import List, Tuple, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))


# ── reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


# ── regime-switching price series ──────────────────────────────────────────────

def generate_prices(n_days: int = 1200, seed: int = 42) -> np.ndarray:
    """Bull / bear / sideways regimes with occasional vol spikes."""
    np.random.seed(seed)
    returns = []
    day = 0
    while day < n_days:
        length = int(np.random.randint(40, 120))
        regime = np.random.choice(["bull", "bear", "sideways"])
        if regime == "bull":
            mu, sigma = float(np.random.uniform(0.0005, 0.002)), float(np.random.uniform(0.008, 0.018))
        elif regime == "bear":
            mu, sigma = float(np.random.uniform(-0.002, -0.0003)), float(np.random.uniform(0.012, 0.025))
        else:
            mu, sigma = float(np.random.uniform(-0.0002, 0.0002)), float(np.random.uniform(0.005, 0.012))
        chunk = np.random.normal(mu, sigma, length).tolist()
        if np.random.rand() < 0.15:
            idx = np.random.randint(0, length)
            chunk[idx] += float(np.random.choice([-1, 1])) * float(np.random.uniform(0.03, 0.08))
        returns.extend(chunk)
        day += length
    prices = 100.0 * np.exp(np.cumsum(np.array(returns[:n_days], dtype=np.float32)))
    return prices.astype(np.float32)


# ── feature engineering ────────────────────────────────────────────────────────

def compute_features(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """5 technical features aligned to the same length."""
    n = len(prices)

    def sma(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    def rsi(arr, period=14):
        deltas = np.diff(arr)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        ag = np.convolve(gains,  np.ones(period) / period, mode="valid")
        al = np.convolve(losses, np.ones(period) / period, mode="valid")
        rs = np.where(al == 0, 100.0, ag / (al + 1e-9))
        return 100 - 100 / (1 + rs)

    sma10 = sma(prices, window)
    rsi14 = rsi(prices, 14)
    min_len = min(len(sma10), len(rsi14))

    feats = []
    for i in range(min_len):
        idx = n - min_len + i
        p   = prices[idx]
        s   = sma10[len(sma10) - min_len + i]
        price_norm = p / s - 1.0 if s > 0 else 0.0
        ret5  = (prices[idx] / prices[max(0, idx - 5)]  - 1.0) if idx >= 5  else 0.0
        ret10 = (prices[idx] / prices[max(0, idx - 10)] - 1.0) if idx >= 10 else 0.0
        chunk = prices[max(0, idx - 5): idx + 1]
        vol5  = float(np.std(chunk)) / (float(np.mean(chunk)) + 1e-9) if len(chunk) > 1 else 0.0
        rsi_v = rsi14[i] / 100.0 - 0.5
        feats.append([price_norm, ret5, ret10, vol5, rsi_v])

    return np.array(feats, dtype=np.float32)


# ── trading environment ────────────────────────────────────────────────────────

class StockEnv:
    """
    Single-asset discrete trading environment.
    Actions: 0 = HOLD, 1 = BUY, 2 = SELL
    State  : 5 technical features + [position, unrealised_pnl%]
    Reward : log-return of portfolio value each step (dense, no double-counting).
    """

    HOLD, BUY, SELL = 0, 1, 2

    def __init__(
        self,
        prices:      np.ndarray,
        features:    np.ndarray,
        cash:        float = 10_000.0,
        trade_cost:  float = 0.002,
    ):
        assert len(prices) == len(features)
        self.prices     = prices
        self.features   = features
        self.init_cash  = cash
        self.trade_cost = trade_cost
        self.state_dim  = features.shape[1] + 2
        self.n_actions  = 3
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_idx  = 0
        self.cash      = self.init_cash
        self.position  = 0
        self.entry     = 0.0
        self.portval   = self.init_cash
        self.trade_log: List[dict] = []
        return self._obs()

    def _obs(self) -> np.ndarray:
        f    = self.features[self.step_idx]
        upnl = 0.0
        if self.position == 1:
            upnl = (self.prices[self.step_idx] - self.entry) / (self.entry + 1e-9)
        return np.concatenate([f, [float(self.position), upnl]], dtype=np.float32)

    # valid actions given current position
    def valid_actions(self) -> List[int]:
        return [self.HOLD, self.BUY]  if self.position == 0 else [self.HOLD, self.SELL]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # mask invalid actions silently to HOLD
        if action == self.BUY  and self.position == 1: action = self.HOLD
        if action == self.SELL and self.position == 0: action = self.HOLD

        price = self.prices[self.step_idx]
        prev  = self.portval

        tc = 0.0
        if action == self.BUY:
            tc          = price * self.trade_cost
            self.entry  = price
            self.cash  -= price + tc
            self.position = 1
            self.trade_log.append({"step": self.step_idx, "action": "BUY",  "price": float(price)})
        elif action == self.SELL:
            tc          = price * self.trade_cost
            pnl         = (price - tc) - self.entry
            self.cash  += price - tc
            self.position = 0
            self.entry    = 0.0
            self.trade_log.append({"step": self.step_idx, "action": "SELL", "price": float(price), "pnl": float(pnl)})

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1

        next_p   = self.prices[min(self.step_idx, len(self.prices) - 1)]
        self.portval = self.cash + next_p * self.position

        # log-return reward (dense, cost baked in)
        reward = float(math.log((self.portval + 1e-9) / (prev + 1e-9)))
        if tc > 0:
            reward -= self.trade_cost * 0.5   # extra churn penalty

        if done and self.position == 1:
            tc2 = next_p * self.trade_cost
            self.cash    += next_p - tc2
            self.position = 0
            self.portval  = self.cash

        return self._obs(), reward, done, {"portval": self.portval}

    def total_return(self) -> float:
        return (self.portval - self.init_cash) / self.init_cash


# ── neural network layers ──────────────────────────────────────────────────────

class DenseLayer:
    """Fully-connected layer: ReLU or linear activation, Adam optimiser."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale       = math.sqrt(2.0 / in_dim)
        self.W      = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b      = np.zeros(out_dim, dtype=np.float32)
        self.act    = activation
        self.dW = self.db = None
        # Adam moments
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self._t = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        self._a = np.maximum(0, self._z) if self.act == "relu" else self._z
        return self._a

    def backward(self, g: np.ndarray) -> np.ndarray:
        if self.act == "relu":
            g = g * (self._z > 0)
        self.dW = self._x.T @ g
        self.db = g.sum(axis=0)
        return g @ self.W.T

    def adam(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        for m, v, g, attr in [(self.mW, self.vW, self.dW, "W"),
                               (self.mb, self.vb, self.db, "b")]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            mh   = m / (1 - beta1 ** t)
            vh   = v / (1 - beta2 ** t)
            if attr == "W":
                self.W -= lr * mh / (np.sqrt(vh) + eps)
            else:
                self.b -= lr * mh / (np.sqrt(vh) + eps)

    def clip_grads(self, max_norm: float = 1.0):
        norm = math.sqrt(float(np.sum(self.dW ** 2) + np.sum(self.db ** 2)))
        if norm > max_norm:
            s = max_norm / norm
            self.dW *= s
            self.db *= s


# ── actor network ──────────────────────────────────────────────────────────────

class ActorNetwork:
    """
    π(a | s; θ_π)  →  softmax probability over n_actions.

    Loss = -Σ log π(aₜ | sₜ) · Aₜ  -  β · H(π)
           ───────────────────────────   ───────────
               policy gradient loss       entropy bonus
    """

    def __init__(self, state_dim: int, n_actions: int = 3, lr: float = 3e-4):
        self.layers   = [
            DenseLayer(state_dim, 128, "relu"),
            DenseLayer(128,        64, "relu"),
            DenseLayer(64, n_actions, "linear"),
        ]
        self.lr        = lr
        self.n_actions = n_actions

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # numerically stable row-wise softmax
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Returns probability distribution π(a|s). Shape: (batch, n_actions)."""
        for layer in self.layers:
            x = layer.forward(x)
        return self._softmax(x)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.forward(state.reshape(1, -1))[0]

    def train(
        self,
        states:     np.ndarray,
        actions:    np.ndarray,
        advantages: np.ndarray,
        entropy_coef: float = 0.01,
    ) -> Tuple[float, float]:
        """
        Policy gradient loss + entropy bonus via REINFORCE-style gradient.

        The gradient of the combined loss w.r.t. the pre-softmax logits is:

          ∂L/∂z_i = π_i * [ -A*(1_{i==a} / π_a)  +  β*(log π_i + 1) ]
                  = -A*(1_{i==a} - π_i)  +  β*π_i*(log π_i + 1 - Σ π_j(log π_j+1))

        Simplification used here: treat as cross-entropy gradient weighted by
        advantage, minus entropy gradient. Each term is derived analytically
        so no numerical differentiation needed.
        """
        probs = self.forward(states)                        # (T, A)
        T     = len(actions)
        idx   = np.arange(T)

        chosen_log = np.log(probs[idx, actions] + 1e-9)    # (T,)
        pg_loss    = -float(np.mean(chosen_log * advantages))

        log_p      = np.log(probs + 1e-9)                  # (T, A)
        entropy    = -float(np.mean(np.sum(probs * log_p, axis=-1)))

        # PG gradient w.r.t. logits: -A * (1_a - π)  / T
        d_pg                = probs.copy()
        d_pg[idx, actions] -= 1.0
        d_pg               *= -(advantages[:, None]) / T

        # Entropy gradient w.r.t. logits: π * (log π + 1) - π * Σ_j π_j*(log π_j+1)
        h                   = log_p + 1.0                   # (T, A)
        d_ent               = probs * (h - (probs * h).sum(axis=-1, keepdims=True))

        # Total: minimise pg_loss, maximise entropy → subtract d_ent
        grad = d_pg - entropy_coef * d_ent

        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        for layer in self.layers:
            layer.clip_grads(1.0)
            layer.adam(self.lr)

        return pg_loss, entropy


# ── critic network ─────────────────────────────────────────────────────────────

class CriticNetwork:
    """
    V(s; θ_v)  →  scalar value estimate.
    Loss = MSE( V(s), returns )
    """

    def __init__(self, state_dim: int, lr: float = 1e-3):
        self.layers = [
            DenseLayer(state_dim, 128, "relu"),
            DenseLayer(128,        64, "relu"),
            DenseLayer(64,          1, "linear"),
        ]
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x                                            # (batch, 1)

    def predict(self, state: np.ndarray) -> float:
        return float(self.forward(state.reshape(1, -1))[0, 0])

    def train(self, states: np.ndarray, returns: np.ndarray) -> float:
        """MSE loss, backprop, Adam. Returns scalar loss."""
        v     = self.forward(states)                        # (T, 1)
        delta = v - returns[:, None]                        # (T, 1)
        loss  = float(np.mean(delta ** 2))

        # Huber-clip for robustness
        dc    = np.clip(delta, -0.1, 0.1) / len(returns)
        g     = dc
        for layer in reversed(self.layers):
            g = layer.backward(g)
        for layer in self.layers:
            layer.clip_grads(1.0)
            layer.adam(self.lr)

        return loss


# ── rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one episode's worth of experience:
      states, actions, rewards, dones, values (from critic)

    Provides:
      compute_returns()   — discounted Monte-Carlo returns
      compute_gae()       — Generalised Advantage Estimation
    """

    def __init__(self):
        self.states:  List[np.ndarray] = []
        self.actions: List[int]        = []
        self.rewards: List[float]      = []
        self.dones:   List[bool]       = []
        self.values:  List[float]      = []

    def push(self, state, action, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Simple discounted returns G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ..."""
        R, returns = 0.0, []
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma * R * (1 - float(d))
            returns.insert(0, R)
        return np.array(returns, dtype=np.float32)

    def compute_gae(
        self,
        gamma:  float = 0.99,
        lam:    float = 0.95,
        last_v: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAE-λ advantage estimation:
          δₜ   = rₜ + γ V(sₜ₊₁) - V(sₜ)
          Aₜ   = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...
          Gₜ   = Aₜ + V(sₜ)          (returns for critic target)

        λ=1  →  Monte-Carlo (low bias, high variance)
        λ=0  →  TD(0)       (high bias, low variance)
        λ=0.95 is the standard sweet spot.
        """
        T      = len(self.rewards)
        advs   = np.zeros(T, dtype=np.float32)
        gae    = 0.0
        values = self.values + [last_v]

        for t in reversed(range(T)):
            next_v  = values[t + 1] * (1 - float(self.dones[t]))
            delta   = self.rewards[t] + gamma * next_v - values[t]
            gae     = delta + gamma * lam * (1 - float(self.dones[t])) * gae
            advs[t] = gae

        returns = advs + np.array(self.values, dtype=np.float32)
        return advs, returns

    def as_arrays(self):
        return (
            np.stack(self.states).astype(np.float32),
            np.array(self.actions, dtype=np.int32),
        )

    def clear(self):
        self.__init__()


# ── A2C agent ──────────────────────────────────────────────────────────────────

class A2CAgent:
    """
    Advantage Actor-Critic (A2C).

    One update per episode (on-policy):
      1. Roll out full episode, storing (s, a, r, done, V(s)).
      2. Compute GAE advantages and discounted returns.
      3. Normalise advantages (mean=0, std=1) for stable gradients.
      4. Update critic: minimise MSE(V(s), returns).
      5. Update actor:  maximise Σ log π(a|s) · A + β · H(π).
    """

    def __init__(
        self,
        state_dim:    int,
        n_actions:    int   = 3,
        actor_lr:     float = 3e-4,
        critic_lr:    float = 1e-3,
        gamma:        float = 0.99,
        lam:          float = 0.95,   # GAE lambda
        entropy_coef: float = 0.01,   # entropy bonus weight
    ):
        self.actor        = ActorNetwork(state_dim, n_actions, actor_lr)
        self.critic       = CriticNetwork(state_dim, critic_lr)
        self.gamma        = gamma
        self.lam          = lam
        self.entropy_coef = entropy_coef
        self.buffer       = RolloutBuffer()
        self.actor_losses:  List[float] = []
        self.critic_losses: List[float] = []
        self.entropies:     List[float] = []

    def select_action(self, state: np.ndarray, position: int = -1) -> Tuple[int, float]:
        """
        Sample from actor distribution, masked to valid actions.
        Returns (action, log_prob).
        """
        probs = self.actor.predict(state).copy()

        # Action masking
        if position == 0:
            probs[StockEnv.SELL] = 0.0
        elif position == 1:
            probs[StockEnv.BUY]  = 0.0
        probs /= probs.sum() + 1e-9

        action = int(np.random.choice(len(probs), p=probs))
        return action, float(math.log(probs[action] + 1e-9))

    def push(self, state, action, reward, done):
        value = self.critic.predict(state)
        self.buffer.push(state, action, reward, done, value)

    def update(self) -> dict:
        """Run one A2C update on the stored episode."""
        states, actions = self.buffer.as_arrays()

        # Compute GAE advantages and critic targets
        last_state = self.buffer.states[-1]
        last_v     = self.critic.predict(last_state) if not self.buffer.dones[-1] else 0.0
        advs, rets = self.buffer.compute_gae(self.gamma, self.lam, last_v)

        # Decay entropy coefficient: high early (explore), low late (exploit)
        ep_count = len(self.actor_losses) + 1
        current_entropy_coef = max(0.005, self.entropy_coef * (0.97 ** ep_count))

        # Normalise advantages — critical for stable actor gradients
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Scale returns for critic (same scale as rewards * 100)
        # Critic update — train on raw returns (same scale as rewards)
        c_loss = self.critic.train(states, rets)

        # Actor update
        a_loss, entropy = self.actor.train(states, actions, advs, current_entropy_coef)

        self.actor_losses.append(a_loss)
        self.critic_losses.append(c_loss)
        self.entropies.append(entropy)

        self.buffer.clear()

        return {
            "actor_loss":  round(a_loss,   6),
            "critic_loss": round(c_loss,   6),
            "entropy":     round(entropy,  4),
        }


# ── training loop ──────────────────────────────────────────────────────────────

def train(
    agent:      A2CAgent,
    env:        StockEnv,
    n_episodes: int  = 150,
    verbose:    bool = True,
) -> List[dict]:
    history = []

    for ep in range(n_episodes):
        state = env.reset()
        done  = False

        while not done:
            action, _ = agent.select_action(state, position=env.position)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, done)
            state = next_state

        metrics = agent.update()
        ret     = env.total_return()
        n_trades = len(env.trade_log)

        record = {
            "episode":     ep + 1,
            "return_pct":  round(ret * 100, 2),
            "n_trades":    n_trades,
            **metrics,
        }
        history.append(record)

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"Ep {ep+1:3d} | return {ret*100:+6.2f}% | "
                f"trades {n_trades:3d} | "
                f"actor {metrics['actor_loss']:+.4f} | "
                f"critic {metrics['critic_loss']:.5f} | "
                f"H {metrics['entropy']:.3f}"
            )

    return history


# ── backtest ───────────────────────────────────────────────────────────────────

def backtest(agent: A2CAgent, env: StockEnv, verbose: bool = True) -> dict:
    """Greedy evaluation: argmax of actor probability distribution."""
    state  = env.reset()
    done   = False
    values = [env.init_cash]
    actions_taken = []

    while not done:
        probs = agent.actor.predict(state).copy()
        if env.position == 0: probs[StockEnv.SELL] = 0.0
        if env.position == 1: probs[StockEnv.BUY]  = 0.0
        probs /= probs.sum() + 1e-9
        # Stochastic policy at test time — argmax collapses all exploration
        # and produces degenerate HOLD-only behaviour when probs are close.
        action = int(np.random.choice(len(probs), p=probs))

        state, _, done, info = env.step(action)
        values.append(info["portval"])
        actions_taken.append(action)

    ret     = env.total_return()
    bnh     = (env.prices[-1] - env.prices[0]) / env.prices[0]
    n_buys  = actions_taken.count(StockEnv.BUY)
    n_sells = actions_taken.count(StockEnv.SELL)

    port_arr = np.array(values, dtype=np.float32)
    daily_r  = np.diff(port_arr) / (port_arr[:-1] + 1e-9)
    sharpe   = float(np.mean(daily_r) / (np.std(daily_r) + 1e-9) * math.sqrt(252))

    peak   = np.maximum.accumulate(port_arr)
    max_dd = float(np.min((port_arr - peak) / (peak + 1e-9)))

    result = {
        "final_portfolio": round(float(values[-1]), 2),
        "total_return_pct": round(ret * 100,         2),
        "bnh_return_pct":   round(bnh * 100,         2),
        "alpha_pct":        round((ret - bnh) * 100, 2),
        "sharpe_ratio":     round(sharpe,             3),
        "max_drawdown_pct": round(max_dd * 100,       2),
        "n_buys":  n_buys,
        "n_sells": n_sells,
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


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_results(history: List[dict], result: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    eps          = [r["episode"]      for r in history]
    returns      = [r["return_pct"]   for r in history]
    trades       = [r["n_trades"]     for r in history]
    actor_losses = [r["actor_loss"]   for r in history]
    entropies    = [r["entropy"]      for r in history]

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("A2C Stock Trader — Training", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(4, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eps, returns, color="#2563EB", linewidth=1.0, alpha=0.7)
    if len(returns) >= 10:
        rm = np.convolve(returns, np.ones(10)/10, mode="valid")
        ax1.plot(eps[9:], rm, color="#DC2626", linewidth=1.8, label="10-ep avg")
        ax1.legend(fontsize=9)
    ax1.axhline(0, color="#94A3B8", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Return %", fontsize=9); ax1.set_title("Episode return", fontsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(eps, trades, color="#7C3AED", alpha=0.55, width=0.8)
    ax2.set_ylabel("# Trades", fontsize=9); ax2.set_title("Trades per episode", fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(eps, actor_losses, color="#D97706", linewidth=1.1)
    ax3.set_ylabel("Actor loss", fontsize=9); ax3.set_title("Policy gradient loss", fontsize=10)
    ax3.grid(True, alpha=0.3, linewidth=0.5)

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(eps, entropies, color="#059669", linewidth=1.1)
    ax4.set_ylabel("H(π)", fontsize=9); ax4.set_title("Policy entropy (exploration)", fontsize=10)
    ax4.set_xlabel("Episode", fontsize=9)
    ax4.grid(True, alpha=0.3, linewidth=0.5)

    fig.savefig(os.path.join(out_dir, "ac_training_curve.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # equity curve
    pv   = result["portfolio_values"]
    n    = len(pv)
    init = pv[0]
    bnh  = np.linspace(init, init * (1 + result["bnh_return_pct"] / 100), n)

    fig2, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n), pv,  color="#2563EB", linewidth=1.5, label="A2C agent")
    ax.plot(range(n), bnh, color="#94A3B8", linewidth=1.2, linestyle="--", label="Buy & hold")
    ax.axhline(init, color="#E5E7EB", linewidth=0.5)
    for t in result.get("trade_log", []):
        x   = t["step"]
        col = "#16A34A" if t["action"] == "BUY" else "#DC2626"
        mk  = "^" if t["action"] == "BUY" else "v"
        if x < n:
            ax.scatter(x, pv[x], color=col, marker=mk, s=55, zorder=5)
    ax.set_title(
        f"Backtest equity   A2C {result['total_return_pct']:+.2f}%   "
        f"B&H {result['bnh_return_pct']:+.2f}%   Alpha {result['alpha_pct']:+.2f}%",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Step", fontsize=9); ax.set_ylabel("Portfolio ($)", fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linewidth=0.5)
    fig2.savefig(os.path.join(out_dir, "ac_equity_curve.png"), dpi=140, bbox_inches="tight")
    plt.close(fig2)

    print("Saved: ac_training_curve.png  ac_equity_curve.png")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    print("Actor-Critic (A2C) Stock Trader")
    print("=" * 50)

    prices   = generate_prices(n_days=2000, seed=42)
    features = compute_features(prices, window=10)
    prices   = prices[-len(features):]

    split      = int(len(prices) * 0.8)
    tr_p, tr_f = prices[:split],  features[:split]
    te_p, te_f = prices[split:],  features[split:]

    print(f"Train steps : {split}  |  Test steps : {len(te_p)}")
    print(f"State dim   : {tr_f.shape[1] + 2}")

    train_env = StockEnv(tr_p, tr_f)
    test_env  = StockEnv(te_p, te_f)

    agent = A2CAgent(
        state_dim    = train_env.state_dim,
        n_actions    = 3,
        actor_lr     = 5e-4,
        critic_lr    = 5e-4,
        gamma        = 0.99,
        lam          = 0.95,
        entropy_coef = 0.05,
    )

    print("\nTraining...\n")
    history = train(agent, train_env, n_episodes=150, verbose=True)

    result = backtest(agent, test_env, verbose=True)

    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.float32, np.float64)): return float(o)
            if isinstance(o, (np.int32,   np.int64)):   return int(o)
            return super().default(o)

    with open(os.path.join(_DIR, "ac_training_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NpEncoder)
    with open(os.path.join(_DIR, "ac_backtest_result.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "portfolio_values"},
                  f, indent=2, cls=NpEncoder)

    plot_results(history, result, _DIR)
    print("\nSaved: ac_training_history.json  ac_backtest_result.json")
    return history, result


if __name__ == "__main__":
    main()