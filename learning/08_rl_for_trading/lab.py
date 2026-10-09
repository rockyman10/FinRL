"""Module 08 lab: reinforcement learning against a known optimum.

Run:  python learning/08_rl_for_trading/lab.py

A single asset has a persistent, noisy return signal and proportional trading
costs. Because we built the market, we can solve for the optimal policy by
dynamic programming and see how close model-free Q-learning gets, how much data
it needs, and how much its results depend on the random seed.

Parts
  A. The model-based optimum (value iteration) versus myopic trading.
  B. Tabular Q-learning: sample efficiency and overfitting.
  C. Seed dispersion: why one RL backtest is not a result.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import TRADING_DAYS, header  # noqa: E402

PHI, SIG_SIGNAL, SIG_NOISE, COST = 0.9, 0.001, 0.01, 0.0010
N_BINS = 7
POSITIONS = np.array([-1, 0, 1])
DISCOUNT = 0.99
EDGES = np.quantile(np.random.default_rng(0).standard_normal(100_000) * SIG_SIGNAL, np.linspace(0, 1, N_BINS + 1)[1:-1])


def simulate(T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(T)
    s[0] = SIG_SIGNAL * rng.standard_normal()
    for t in range(1, T):
        s[t] = PHI * s[t - 1] + SIG_SIGNAL * np.sqrt(1 - PHI**2) * rng.standard_normal()
    r_next = s + SIG_NOISE * rng.standard_normal(T)  # return earned from t to t+1
    return np.digitize(s, EDGES), r_next


def reward(pos_prev: int, pos_new: int, r: float) -> float:
    return POSITIONS[pos_new] * r - COST * abs(POSITIONS[pos_new] - POSITIONS[pos_prev])


def evaluate(policy: np.ndarray, bins: np.ndarray, r_next: np.ndarray) -> float:
    """Annualised Sharpe of a policy[bin, current position index] -> new position index."""
    pos, pnl = 1, np.empty(len(bins))  # start flat (index 1 = position 0)
    for t in range(len(bins)):
        new = policy[bins[t], pos]
        pnl[t] = reward(pos, new, r_next[t])
        pos = new
    return float(np.sqrt(TRADING_DAYS) * pnl.mean() / pnl.std())


def value_iteration(P: np.ndarray, mean_r: np.ndarray) -> np.ndarray:
    """Q*(bin, pos, new_pos) for the known (estimated) Markov chain of signal bins."""
    V = np.zeros((N_BINS, 3))
    for _ in range(2000):
        Q = np.empty((N_BINS, 3, 3))
        for a in range(3):
            immediate = POSITIONS[a] * mean_r[:, None] - COST * np.abs(POSITIONS[a] - POSITIONS[None, :])
            Q[:, :, a] = immediate + DISCOUNT * (P @ V[:, a])[:, None]
        V_new = Q.max(axis=2)
        if np.max(np.abs(V_new - V)) < 1e-12:
            break
        V = V_new
    return Q.argmax(axis=2)


def q_learning(bins: np.ndarray, r_next: np.ndarray, epochs: int, rng: np.random.Generator) -> np.ndarray:
    Q = np.zeros((N_BINS, 3, 3))
    visits = np.zeros_like(Q)
    for epoch in range(epochs):
        eps = max(0.05, 1.0 - epoch / (0.7 * epochs))
        pos = 1
        for t in range(len(bins) - 1):
            b = bins[t]
            a = rng.integers(3) if rng.random() < eps else int(Q[b, pos].argmax())
            rwd = reward(pos, a, r_next[t])
            visits[b, pos, a] += 1
            lr = 1.0 / visits[b, pos, a] ** 0.6
            target = rwd + DISCOUNT * Q[bins[t + 1], a].max()
            Q[b, pos, a] += lr * (target - Q[b, pos, a])
            pos = a
    return Q.argmax(axis=2)


rng = np.random.default_rng(88)
test_bins, test_r = simulate(TRADING_DAYS * 50, rng)

# ---------------------------------------------------------------------------
header("A. The optimum when the model is known")
# ---------------------------------------------------------------------------
big_bins, big_r = simulate(2_000_000, rng)
P = np.zeros((N_BINS, N_BINS))
np.add.at(P, (big_bins[:-1], big_bins[1:]), 1)
P /= P.sum(axis=1, keepdims=True)
mean_r = np.array([big_r[big_bins == k].mean() for k in range(N_BINS)])
optimal = value_iteration(P, mean_r)
myopic = np.tile(np.where(mean_r > 0, 2, 0)[:, None], (1, 3))  # sign of E[r], ignores costs
always_long = np.full((N_BINS, 3), 2)

print("Optimal new position by signal bin (rows) and current position (columns):")
print(pd.DataFrame(POSITIONS[optimal], columns=["from -1", "from 0", "from +1"]).to_string())
results = {
    "always long": evaluate(always_long, test_bins, test_r),
    "myopic sign(E[r]), cost-blind": evaluate(myopic, test_bins, test_r),
    "dynamic programming optimum": evaluate(optimal, test_bins, test_r),
}
for k, v in results.items():
    print(f"  {k:32s} out-of-sample Sharpe {v:5.2f}")
print("The optimum keeps its position in the middle bins: a no-trade region, as in Module 06.")


# ---------------------------------------------------------------------------
header("B. Q-learning from data: how much history is enough?")
# ---------------------------------------------------------------------------
rows = []
for years in [1, 3, 10, 30]:
    train_bins, train_r = simulate(TRADING_DAYS * years, rng)
    pol = q_learning(train_bins, train_r, epochs=max(3, int(60 / years)), rng=rng)
    rows.append(
        {
            "training years": years,
            "in-sample Sharpe": evaluate(pol, train_bins, train_r),
            "out-of-sample Sharpe": evaluate(pol, test_bins, test_r),
            "agrees with optimum": (pol == optimal).mean(),
        }
    )
table_b = pd.DataFrame(rows).set_index("training years")
print(table_b.round(2).to_string())
print(f"(optimum out of sample: {results['dynamic programming optimum']:.2f})")
print(
    "With little data the agent fits noise: high in-sample, poor out-of-sample. FinRL's\n"
    "Dow 30 examples train on about 6.5 years (config.py:10-11) with a much larger state."
)


# ---------------------------------------------------------------------------
header("C. Same data, different seeds")
# ---------------------------------------------------------------------------
train_bins, train_r = simulate(TRADING_DAYS * 3, np.random.default_rng(1234))
oos = [
    evaluate(q_learning(train_bins, train_r, epochs=20, rng=np.random.default_rng(seed)), test_bins, test_r)
    for seed in range(12)
]
print(f"out-of-sample Sharpe over 12 seeds: min {min(oos):.2f}, median {np.median(oos):.2f}, max {max(oos):.2f}")
print(
    "Reporting the best seed is a hidden multiple test (Module 05). Henderson et al.\n"
    "(2018) show the same dispersion for deep RL; report the distribution."
)
assert results["dynamic programming optimum"] > results["myopic sign(E[r]), cost-blind"]
assert table_b.loc[30, "out-of-sample Sharpe"] > table_b.loc[1, "out-of-sample Sharpe"]
print("\nAll Module 08 checks passed.")
