"""
Proximal Policy Optimisation (PPO) — Stock Trading Agent
=========================================================
Pure NumPy, no PyTorch/TensorFlow.

What PPO adds over A2C
-----------------------
  1. Clipped surrogate objective
       L_CLIP = E[ min(r_t·A_t,  clip(r_t, 1-ε, 1+ε)·A_t) ]
       where r_t = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
       The clip prevents any single update from moving the policy too far.

  2. Multiple minibatch epochs per rollout
       A2C does one update per episode. PPO shuffles the rollout into
       minibatches and runs K epochs of gradient descent — much better
       sample efficiency without the instability of large policy steps.

  3. KL early stopping
       If the mean KL divergence between new and old policy exceeds a
       threshold, stop the epoch early. Belt-and-suspenders alongside clip.

  4. Value function clipping
       Clip critic loss to prevent large value updates destabilising
       the advantage baseline.

Architecture
------------
  ActorNetwork   — π(a|s; θ_π): softmax over {HOLD, BUY, SELL}
  CriticNetwork  — V(s; θ_v): scalar state value
  PPOAgent       — rollout buffer, GAE, clipped PPO update loop
  StockEnv       — regime-switching single-asset trading environment
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


# ── regime-switching price data ────────────────────────────────────────────────

def generate_prices(n_days: int = 2000, seed: int = 42) -> np.ndarray:
    """Bull / bear / sideways regimes with vol spikes."""
    np.random.seed(seed)
    returns, day = [], 0
    while day < n_days:
        length = int(np.random.randint(40, 120))
        regime = np.random.choice(["bull", "bear", "sideways"])
        if regime == "bull":
            mu, sigma = float(np.random.uniform(0.0005, 0.002)), \
                        float(np.random.uniform(0.008, 0.018))
        elif regime == "bear":
            mu, sigma = float(np.random.uniform(-0.002, -0.0003)), \
                        float(np.random.uniform(0.012, 0.025))
        else:
            mu, sigma = float(np.random.uniform(-0.0002, 0.0002)), \
                        float(np.random.uniform(0.005, 0.012))
        chunk = np.random.normal(mu, sigma, length).tolist()
        if np.random.rand() < 0.15:          # occasional spike
            idx = np.random.randint(0, length)
            chunk[idx] += float(np.random.choice([-1, 1])) * \
                          float(np.random.uniform(0.03, 0.08))
        returns.extend(chunk)
        day += length
    prices = 100.0 * np.exp(np.cumsum(np.array(returns[:n_days], dtype=np.float32)))
    return prices.astype(np.float32)


# ── feature engineering ────────────────────────────────────────────────────────

def compute_features(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """5 technical features aligned to common length."""
    n = len(prices)

    def sma(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    def rsi(arr, p=14):
        d = np.diff(arr)
        g = np.where(d > 0, d, 0.0)
        l = np.where(d < 0, -d, 0.0)
        ag = np.convolve(g, np.ones(p) / p, mode="valid")
        al = np.convolve(l, np.ones(p) / p, mode="valid")
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
    Single-asset, discrete-action trading environment.
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward : log-return of portfolio value (dense, costs baked in).
    """
    HOLD, BUY, SELL = 0, 1, 2

    def __init__(self, prices: np.ndarray, features: np.ndarray,
                 cash: float = 10_000.0, trade_cost: float = 0.002):
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
        upnl = ((self.prices[self.step_idx] - self.entry) / (self.entry + 1e-9)
                 if self.position else 0.0)
        return np.concatenate([f, [float(self.position), upnl]], dtype=np.float32)

    def valid_actions(self) -> List[int]:
        return [self.HOLD, self.BUY] if self.position == 0 else [self.HOLD, self.SELL]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action == self.BUY  and self.position == 1: action = self.HOLD
        if action == self.SELL and self.position == 0: action = self.HOLD

        price = self.prices[self.step_idx]
        prev  = self.portval
        tc    = 0.0

        if action == self.BUY:
            tc = price * self.trade_cost
            self.entry = price
            self.cash -= price + tc
            self.position = 1
            self.trade_log.append({"step": self.step_idx, "action": "BUY",
                                   "price": float(price)})
        elif action == self.SELL:
            tc  = price * self.trade_cost
            pnl = (price - tc) - self.entry
            self.cash += price - tc
            self.position = 0
            self.entry = 0.0
            self.trade_log.append({"step": self.step_idx, "action": "SELL",
                                   "price": float(price), "pnl": float(pnl)})

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1
        nxt  = self.prices[min(self.step_idx, len(self.prices) - 1)]
        self.portval = self.cash + nxt * self.position

        reward = float(math.log((self.portval + 1e-9) / (prev + 1e-9)))
        if tc > 0:
            reward -= self.trade_cost * 0.5   # extra churn penalty

        if done and self.position == 1:
            tc2 = nxt * self.trade_cost
            self.cash += nxt - tc2
            self.position = 0
            self.portval  = self.cash

        return self._obs(), reward, done

    def total_return(self) -> float:
        return (self.portval - self.init_cash) / self.init_cash


# ── neural network layers ──────────────────────────────────────────────────────

class DenseLayer:
    """FC layer with ReLU/linear activation and Adam optimiser."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale    = math.sqrt(2.0 / in_dim)
        self.W   = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b   = np.zeros(out_dim, dtype=np.float32)
        self.act = activation
        self.dW  = self.db = None
        self.mW  = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb  = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self._t  = 0

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

    def adam(self, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        for m, v, g, attr in [(self.mW, self.vW, self.dW, "W"),
                               (self.mb, self.vb, self.db, "b")]:
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g ** 2
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            if attr == "W": self.W -= lr * mh / (np.sqrt(vh) + eps)
            else:           self.b -= lr * mh / (np.sqrt(vh) + eps)

    def clip_grads(self, norm: float = 0.5):
        n = math.sqrt(float(np.sum(self.dW ** 2) + np.sum(self.db ** 2)))
        if n > norm:
            s = norm / n
            self.dW *= s; self.db *= s

    def get_params(self):
        return (self.W.copy(), self.b.copy())

    def set_params(self, params):
        self.W[:], self.b[:] = params


# ── actor network ──────────────────────────────────────────────────────────────

class ActorNetwork:
    """
    π(a|s; θ)  →  softmax probabilities over n_actions.

    PPO-specific additions vs A2C actor:
      • get_log_probs()   — returns log π(a|s) for stored old policy
      • snapshot()        — copies weights to θ_old before each update epoch
      • kl_from_old()     — computes mean KL(π_old || π_new) for early stopping
    """

    def __init__(self, state_dim: int, n_actions: int = 3, lr: float = 3e-4):
        self.layers   = [
            DenseLayer(state_dim, 128, "relu"),
            DenseLayer(128,        64, "relu"),
            DenseLayer(64, n_actions, "linear"),
        ]
        self.lr        = lr
        self.n_actions = n_actions
        self._old_params: Optional[list] = None  # snapshot of θ_old

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return self._softmax(x)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.forward(state.reshape(1, -1))[0]

    # ── PPO-specific helpers ────────────────────────────────────────────────────

    def snapshot(self):
        """Save current weights as θ_old (called once before K epochs)."""
        self._old_params = [l.get_params() for l in self.layers]

    def forward_old(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through θ_old without disturbing current weights."""
        assert self._old_params is not None, "Call snapshot() first"
        # Temporarily swap weights
        cur_params = [l.get_params() for l in self.layers]
        for l, p in zip(self.layers, self._old_params):
            l.set_params(p)
        out = self.forward(x)
        for l, p in zip(self.layers, cur_params):
            l.set_params(p)
        return out

    def get_log_probs(self, states: np.ndarray, actions: np.ndarray,
                      use_old: bool = False) -> np.ndarray:
        """log π(a|s) for selected actions. Shape: (T,)."""
        probs = self.forward_old(states) if use_old else self.forward(states)
        idx   = np.arange(len(actions))
        return np.log(probs[idx, actions] + 1e-9)

    def kl_from_old(self, states: np.ndarray) -> float:
        """
        Mean KL divergence KL(π_old || π_new).
        KL(p||q) = Σ p * log(p/q)
        Used to decide whether to stop the update epoch early.
        """
        p_old = self.forward_old(states)
        p_new = self.forward(states)
        kl    = np.sum(p_old * (np.log(p_old + 1e-9) - np.log(p_new + 1e-9)), axis=-1)
        return float(np.mean(kl))

    # ── PPO clipped surrogate loss ──────────────────────────────────────────────

    def train_ppo(
        self,
        states:       np.ndarray,   # (B, state_dim)
        actions:      np.ndarray,   # (B,) int
        advantages:   np.ndarray,   # (B,) normalised
        log_probs_old: np.ndarray,  # (B,) log π_old(a|s)
        clip_eps:     float = 0.2,
        entropy_coef: float = 0.01,
    ) -> Tuple[float, float, float]:
        """
        Clipped PPO surrogate loss:

          r_t     = exp(log π_new − log π_old)   (probability ratio)
          L_CLIP  = −mean[ min(r_t·A_t,  clip(r_t, 1−ε, 1+ε)·A_t) ]

        The min + clip combination means:
          • If A > 0 (good action): we want r_t > 1 (increase prob), but clip
            at 1+ε so we never overshoot in one step.
          • If A < 0 (bad action): we want r_t < 1 (decrease prob), but clip
            at 1−ε so we never overshoot in the other direction.

        Returns (pg_loss, entropy, mean_ratio).
        """
        probs    = self.forward(states)             # (B, A)
        B        = len(actions)
        idx      = np.arange(B)

        log_new  = np.log(probs[idx, actions] + 1e-9)   # (B,)
        ratio    = np.exp(log_new - log_probs_old)       # r_t = π_new / π_old

        # Clipped surrogate
        surr1    = ratio * advantages
        surr2    = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        pg_loss  = -float(np.mean(np.minimum(surr1, surr2)))

        # Entropy bonus
        log_p    = np.log(probs + 1e-9)
        entropy  = -float(np.mean(np.sum(probs * log_p, axis=-1)))

        # Gradient of combined loss w.r.t. logits
        # Gradient of −min(surr1, surr2) w.r.t. log_new:
        #   which surrogate is active determines the gradient
        active   = np.where(surr1 <= surr2, ratio, np.clip(ratio, 1-clip_eps, 1+clip_eps))
        d_logits = probs.copy()
        d_logits[idx, actions] -= 1.0               # log-prob → softmax gradient
        d_logits *= -(active * advantages)[:, None] / B

        # Entropy gradient
        h        = log_p + 1.0
        d_ent    = probs * (h - (probs * h).sum(axis=-1, keepdims=True))
        grad     = d_logits - entropy_coef * d_ent

        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        for layer in self.layers:
            layer.clip_grads(0.5)
            layer.adam(self.lr)

        return pg_loss, entropy, float(np.mean(ratio))


# ── critic network ─────────────────────────────────────────────────────────────

class CriticNetwork:
    """
    V(s; θ_v)  →  scalar value.

    PPO value clipping:
      L_V = mean[ max( (V_new−G)², (clip(V_new, V_old−ε, V_old+ε)−G)² ) ]

    This prevents large value updates that would destabilise the advantage
    baseline. V_old is the value prediction from before this round of updates.
    """

    def __init__(self, state_dim: int, lr: float = 5e-4):
        self.layers = [
            DenseLayer(state_dim, 128, "relu"),
            DenseLayer(128,        64, "relu"),
            DenseLayer(64,          1, "linear"),
        ]
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x           # (B, 1)

    def predict(self, state: np.ndarray) -> float:
        return float(self.forward(state.reshape(1, -1))[0, 0])

    def train_ppo(
        self,
        states:    np.ndarray,   # (B, state_dim)
        returns:   np.ndarray,   # (B,) target G_t
        values_old: np.ndarray,  # (B,) V(s) from before this epoch
        clip_eps:  float = 0.2,
    ) -> float:
        """
        Clipped value loss. Returns scalar loss.
        """
        v_new   = self.forward(states)[:, 0]         # (B,)
        v_clip  = np.clip(v_new,
                          values_old - clip_eps,
                          values_old + clip_eps)
        loss1   = (v_new  - returns) ** 2
        loss2   = (v_clip - returns) ** 2
        loss    = float(np.mean(np.maximum(loss1, loss2)))

        # Gradient of max(loss1, loss2): pick the active branch
        active  = np.where(loss1 >= loss2, v_new - returns, v_clip - returns)
        d       = (2 * active / len(returns)).reshape(-1, 1)   # (B, 1)

        g = d
        for layer in reversed(self.layers):
            g = layer.backward(g)
        for layer in self.layers:
            layer.clip_grads(0.5)
            layer.adam(self.lr)

        return loss


# ── rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Collects one full episode (on-policy rollout).
    Provides GAE advantages, returns, and stored log-probs for the PPO ratio.
    """

    def __init__(self):
        self.states:    List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.rewards:   List[float]      = []
        self.dones:     List[bool]       = []
        self.values:    List[float]      = []
        self.log_probs: List[float]      = []   # log π_old(a|s) at collection time

    def push(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95,
                    last_v: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """GAE-λ advantages and discounted returns."""
        T    = len(self.rewards)
        advs = np.zeros(T, dtype=np.float32)
        gae  = 0.0
        vals = self.values + [last_v]
        for t in reversed(range(T)):
            nxt   = vals[t + 1] * (1 - float(self.dones[t]))
            delta = self.rewards[t] + gamma * nxt - vals[t]
            gae   = delta + gamma * lam * (1 - float(self.dones[t])) * gae
            advs[t] = gae
        returns = advs + np.array(self.values, dtype=np.float32)
        return advs, returns

    def as_arrays(self):
        return (np.stack(self.states).astype(np.float32),
                np.array(self.actions,   dtype=np.int32),
                np.array(self.log_probs, dtype=np.float32),
                np.array(self.values,    dtype=np.float32))

    def clear(self):
        self.__init__()


# ── PPO agent ──────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimisation.

    Per-episode update procedure:
      1. Roll out one episode, storing (s, a, r, done, V(s), log π(a|s)).
      2. Snapshot θ_old from current actor weights.
      3. Compute GAE advantages + returns. Normalise advantages.
      4. For K epochs:
           Shuffle rollout into minibatches of size M.
           For each minibatch:
             a. Compute probability ratio r_t = π_new / π_old.
             b. Compute clipped PPO actor loss.
             c. Compute clipped PPO critic loss.
             d. Gradient step on both networks.
           Check KL(π_old || π_new). If > kl_target: break early.
      5. Clear buffer.
    """

    def __init__(
        self,
        state_dim:    int,
        n_actions:    int   = 3,
        actor_lr:     float = 3e-4,
        critic_lr:    float = 5e-4,
        gamma:        float = 0.99,
        lam:          float = 0.95,
        clip_eps:     float = 0.2,      # PPO clip range
        entropy_coef: float = 0.01,
        n_epochs:     int   = 4,        # K gradient epochs per rollout
        minibatch:    int   = 64,       # minibatch size
        kl_target:    float = 0.02,     # KL early-stop threshold
    ):
        self.actor        = ActorNetwork(state_dim, n_actions, actor_lr)
        self.critic       = CriticNetwork(state_dim, critic_lr)
        self.gamma        = gamma
        self.lam          = lam
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.minibatch    = minibatch
        self.kl_target    = kl_target
        self.buffer       = RolloutBuffer()

        # running metrics
        self.actor_losses:  List[float] = []
        self.critic_losses: List[float] = []
        self.entropies:     List[float] = []
        self.kl_divs:       List[float] = []
        self.ratios:        List[float] = []
        self._ep_count:     int         = 0

    def select_action(self, state: np.ndarray,
                      position: int = -1) -> Tuple[int, float, float]:
        """
        Sample from masked actor distribution with epsilon-greedy warmup.
        During the first warmup_eps episodes, inject random exploration so
        the actor sees diverse (s, a, r) tuples before the policy gradient
        locks in. After warmup, pure policy sampling.
        Returns (action, log_prob, value).
        """
        probs = self.actor.predict(state).copy()
        if position == 0: probs[StockEnv.SELL] = 0.0
        if position == 1: probs[StockEnv.BUY]  = 0.0
        probs /= probs.sum() + 1e-9

        # Warmup exploration: linearly decay from eps_start to 0 over warmup_eps
        eps_start   = 0.8
        warmup_eps  = 30
        explore_eps = max(0.0, eps_start * (1 - self._ep_count / warmup_eps))

        if np.random.rand() < explore_eps:
            valid = ([StockEnv.HOLD, StockEnv.BUY]  if position == 0 else
                     [StockEnv.HOLD, StockEnv.SELL] if position == 1 else
                     list(range(self.actor.n_actions)))
            action = int(np.random.choice(valid))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        log_prob = float(math.log(probs[action] + 1e-9))
        value    = self.critic.predict(state)
        return action, log_prob, value

    def push(self, state, action, reward, done, value, log_prob):
        self.buffer.push(state, action, reward, done, value, log_prob)

    def update(self) -> dict:
        """Run PPO update. Returns dict of metrics."""
        self._ep_count += 1

        states, actions, log_probs_old, values_old = self.buffer.as_arrays()

        last_v = (self.critic.predict(self.buffer.states[-1])
                  if not self.buffer.dones[-1] else 0.0)
        advs, rets = self.buffer.compute_gae(self.gamma, self.lam, last_v)

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Decay entropy coefficient
        ent_coef = max(0.005, self.entropy_coef * (0.995 ** self._ep_count))

        # Snapshot θ_old before any gradient steps this round
        self.actor.snapshot()

        T            = len(states)
        ep_a_losses  = []
        ep_c_losses  = []
        ep_entropies = []
        ep_ratios    = []
        epoch_stopped = 0

        for epoch in range(self.n_epochs):
            # Shuffle indices for minibatch sampling
            idxs = np.random.permutation(T)

            for start in range(0, T, self.minibatch):
                mb_idx = idxs[start: start + self.minibatch]
                if len(mb_idx) < 4:     # skip tiny tail batches
                    continue

                mb_s   = states[mb_idx]
                mb_a   = actions[mb_idx]
                mb_adv = advs[mb_idx]
                mb_ret = rets[mb_idx]
                mb_lp  = log_probs_old[mb_idx]
                mb_v   = values_old[mb_idx]

                a_loss, ent, ratio = self.actor.train_ppo(
                    mb_s, mb_a, mb_adv, mb_lp,
                    self.clip_eps, ent_coef,
                )
                c_loss = self.critic.train_ppo(
                    mb_s, mb_ret, mb_v, self.clip_eps,
                )

                ep_a_losses.append(a_loss)
                ep_c_losses.append(c_loss)
                ep_entropies.append(ent)
                ep_ratios.append(ratio)

            # KL early-stop check (once per epoch, on full rollout)
            kl = self.actor.kl_from_old(states)
            if kl > self.kl_target:
                epoch_stopped = epoch + 1
                break

        self.buffer.clear()

        avg_a  = float(np.mean(ep_a_losses))   if ep_a_losses  else 0.0
        avg_c  = float(np.mean(ep_c_losses))   if ep_c_losses  else 0.0
        avg_h  = float(np.mean(ep_entropies))  if ep_entropies else 0.0
        avg_r  = float(np.mean(ep_ratios))     if ep_ratios    else 1.0

        self.actor_losses.append(avg_a)
        self.critic_losses.append(avg_c)
        self.entropies.append(avg_h)
        self.ratios.append(avg_r)

        return {
            "actor_loss":   round(avg_a, 6),
            "critic_loss":  round(avg_c, 6),
            "entropy":      round(avg_h, 4),
            "mean_ratio":   round(avg_r, 4),
            "epochs_run":   epoch_stopped if epoch_stopped else self.n_epochs,
        }


# ── training loop ──────────────────────────────────────────────────────────────

def train(agent: PPOAgent, env: StockEnv,
          n_episodes: int = 200, verbose: bool = True,
          rollout_eps: int = 4) -> List[dict]:
    """
    PPO training with multi-episode rollouts.
    Accumulate rollout_eps episodes before each PPO update.
    This ensures the policy has drifted enough between collection
    and update for the clipped surrogate to have a non-trivial effect.
    """
    history = []
    ep = 0
    while ep < n_episodes:
        # Accumulate rollout_eps episodes into the buffer
        ep_returns = []
        ep_trades  = []
        for _ in range(rollout_eps):
            if ep >= n_episodes:
                break
            state = env.reset()
            done  = False
            while not done:
                action, lp, val = agent.select_action(state, position=env.position)
                next_state, reward, done = env.step(action)
                agent.push(state, action, reward, done, val, lp)
                state = next_state
            ep_returns.append(env.total_return())
            ep_trades.append(len(env.trade_log))
            ep += 1

        # One PPO update on the accumulated buffer
        metrics = agent.update()
        avg_ret = float(np.mean(ep_returns))
        avg_trd = int(np.mean(ep_trades))

        for i, (ret, trd) in enumerate(zip(ep_returns, ep_trades)):
            record = {
                "episode":    ep - len(ep_returns) + i + 1,
                "return_pct": round(ret * 100, 2),
                "n_trades":   trd,
                **metrics,
            }
            history.append(record)

        if verbose and ep % (rollout_eps * 5) < rollout_eps:
            print(
                f"Ep {ep:3d} | return {avg_ret*100:+6.2f}% | "
                f"trades {avg_trd:3d} | "
                f"ratio {metrics['mean_ratio']:.3f} | "
                f"epochs {metrics['epochs_run']} | "
                f"H {metrics['entropy']:.3f}"
            )
    return history


# ── backtest ───────────────────────────────────────────────────────────────────

def backtest(agent: PPOAgent, env: StockEnv, verbose: bool = True) -> dict:
    state  = env.reset()
    done   = False
    values = [env.init_cash]
    acts   = []

    while not done:
        probs = agent.actor.predict(state).copy()
        if env.position == 0: probs[StockEnv.SELL] = 0.0
        if env.position == 1: probs[StockEnv.BUY]  = 0.0
        probs /= probs.sum() + 1e-9
        action = int(np.random.choice(len(probs), p=probs))
        state, _, done = env.step(action)
        values.append(env.portval)
        acts.append(action)

    ret    = env.total_return()
    bnh    = (env.prices[-1] - env.prices[0]) / env.prices[0]
    pa     = np.array(values, dtype=np.float32)
    dr     = np.diff(pa) / (pa[:-1] + 1e-9)
    sharpe = float(np.mean(dr) / (np.std(dr) + 1e-9) * math.sqrt(252))
    peak   = np.maximum.accumulate(pa)
    maxdd  = float(np.min((pa - peak) / (peak + 1e-9)))

    result = {
        "final_portfolio":  round(float(values[-1]), 2),
        "total_return_pct": round(ret * 100, 2),
        "bnh_return_pct":   round(bnh * 100, 2),
        "alpha_pct":        round((ret - bnh) * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "max_drawdown_pct": round(maxdd * 100, 2),
        "n_buys":           acts.count(StockEnv.BUY),
        "n_sells":          acts.count(StockEnv.SELL),
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
        print(f"  Trades (B/S)    : {result['n_buys']} / {result['n_sells']}")
        print("────────────────────────────────────────────────────")
    return result


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_results(history: List[dict], result: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    eps     = [r["episode"]     for r in history]
    rets    = [r["return_pct"]  for r in history]
    trades  = [r["n_trades"]    for r in history]
    ratios  = [r["mean_ratio"]  for r in history]
    entropy = [r["entropy"]     for r in history]
    epochs  = [r["epochs_run"]  for r in history]

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle("PPO Stock Trader — Training", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(5, 1, hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eps, rets, color="#2563EB", linewidth=1.0, alpha=0.7)
    if len(rets) >= 10:
        rm = np.convolve(rets, np.ones(10)/10, mode="valid")
        ax1.plot(eps[9:], rm, color="#DC2626", linewidth=1.8, label="10-ep avg")
        ax1.legend(fontsize=9)
    ax1.axhline(0, color="#94A3B8", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Return %", fontsize=9)
    ax1.set_title("Episode return", fontsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(eps, trades, color="#7C3AED", alpha=0.55, width=0.8)
    ax2.set_ylabel("# Trades", fontsize=9)
    ax2.set_title("Trades per episode", fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(eps, ratios, color="#D97706", linewidth=1.1)
    ax3.axhline(1.0, color="#94A3B8", linewidth=0.5, linestyle="--")
    ax3.axhline(1 + 0.2, color="#E24B4A", linewidth=0.5, linestyle=":")
    ax3.axhline(1 - 0.2, color="#E24B4A", linewidth=0.5, linestyle=":")
    ax3.set_ylabel("Mean ratio r_t", fontsize=9)
    ax3.set_title("PPO probability ratio  (clip bounds ±0.2)", fontsize=10)
    ax3.grid(True, alpha=0.3, linewidth=0.5)

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(eps, entropy, color="#059669", linewidth=1.1)
    ax4.set_ylabel("H(π)", fontsize=9)
    ax4.set_title("Policy entropy", fontsize=10)
    ax4.grid(True, alpha=0.3, linewidth=0.5)

    ax5 = fig.add_subplot(gs[4])
    ax5.bar(eps, epochs, color="#185FA5", alpha=0.6, width=0.8)
    ax5.axhline(4, color="#E24B4A", linewidth=0.7, linestyle="--", label="max epochs")
    ax5.legend(fontsize=9)
    ax5.set_ylabel("Epochs run", fontsize=9)
    ax5.set_xlabel("Episode", fontsize=9)
    ax5.set_title("Epochs per update  (early stop = KL exceeded)", fontsize=10)
    ax5.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    fig.savefig(os.path.join(out_dir, "ppo_training_curve.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Equity curve
    pv   = result["portfolio_values"]
    n    = len(pv)
    bnh  = np.linspace(pv[0], pv[0] * (1 + result["bnh_return_pct"] / 100), n)

    fig2, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n), pv,  color="#2563EB", linewidth=1.5, label="PPO agent")
    ax.plot(range(n), bnh, color="#94A3B8",  linewidth=1.2, linestyle="--", label="Buy & hold")
    ax.axhline(pv[0], color="#E5E7EB", linewidth=0.5)
    for t in result.get("trade_log", []):
        x   = t["step"]
        col = "#16A34A" if t["action"] == "BUY" else "#DC2626"
        mk  = "^" if t["action"] == "BUY" else "v"
        if x < n:
            ax.scatter(x, pv[x], color=col, marker=mk, s=55, zorder=5)
    ax.set_title(
        f"PPO backtest   {result['total_return_pct']:+.2f}%   "
        f"B&H {result['bnh_return_pct']:+.2f}%   Alpha {result['alpha_pct']:+.2f}%",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Portfolio ($)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig2.savefig(os.path.join(out_dir, "ppo_equity_curve.png"), dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: ppo_training_curve.png  ppo_equity_curve.png")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    print("PPO Stock Trader")
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

    agent = PPOAgent(
        state_dim    = train_env.state_dim,
        n_actions    = 3,
        actor_lr     = 3e-4,
        critic_lr    = 5e-4,
        gamma        = 0.99,
        lam          = 0.95,
        clip_eps     = 0.2,
        entropy_coef = 0.05,
        n_epochs     = 4,
        minibatch    = 128,
        kl_target    = 0.02,
    )

    print("\nTraining...\n")
    history = train(agent, train_env, n_episodes=200, verbose=True, rollout_eps=4)

    result = backtest(agent, test_env, verbose=True)

    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.float32, np.float64)): return float(o)
            if isinstance(o, (np.int32,   np.int64)):   return int(o)
            return super().default(o)

    with open(os.path.join(_DIR, "ppo_training_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NpEnc)
    with open(os.path.join(_DIR, "ppo_backtest_result.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "portfolio_values"},
                  f, indent=2, cls=NpEnc)

    plot_results(history, result, _DIR)
    print("\nSaved: ppo_training_history.json  ppo_backtest_result.json")
    return history, result


if __name__ == "__main__":
    main()