"""Module 03 lab: mean-variance theory, estimation error and FinRL's softmax weights.

Run:  python learning/03_portfolio_theory/lab.py

Parts
  A. The efficient frontier in closed form, using the true parameters.
  B. Out of sample: 1/N against estimated mean-variance (DeMiguel et al. 2009).
  C. What FinRL's portfolio environment can actually express.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import TRADING_DAYS, header, simulate_market  # noqa: E402



def frontier_constants(mu, cov):
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    return inv, A, B, C


def gmv(cov):
    w = np.linalg.solve(cov, np.ones(len(cov)))
    return w / w.sum()


def tangency(mu_excess, cov):
    w = np.linalg.solve(cov, mu_excess)
    return w / w.sum()


def ledoit_wolf(X):
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    m = np.trace(S) / N
    d2 = np.linalg.norm(S - m * np.eye(N), "fro") ** 2 / N
    b2 = min(sum(np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in Xc) / T**2 / N, d2)
    delta = b2 / d2
    return delta * m * np.eye(N) + (1 - delta) * S


def softmax(a):
    """FinRL's StockPortfolioEnv.softmax_normalization (env_portfolio.py:225-229)."""
    e = np.exp(a)
    return e / e.sum()


# ---------------------------------------------------------------------------
header("A. The frontier with known parameters")
# ---------------------------------------------------------------------------
N = 30
m = simulate_market(n_assets=N, seed=21, p_calm_to_crisis=0.0)
cov = m.true_cov_calm() * TRADING_DAYS
mu_f = np.array([0.07, 0.02, 0.03])
mu_excess = m.alphas.to_numpy() * TRADING_DAYS + m.betas.to_numpy() @ mu_f
rf = m.rf * TRADING_DAYS

inv, A, B, C = frontier_constants(mu_excess, cov)
D = A * C - B**2
print("Frontier in (sigma^2, mu) space: sigma^2(mu) = (A mu^2 - 2 B mu + C) / D")
w_g, w_t = gmv(cov), tangency(mu_excess, cov)
sr = lambda w: (w @ mu_excess) / np.sqrt(w @ cov @ w)  # noqa: E731
print(f"GMV       : excess mean {w_g @ mu_excess:6.2%}, vol {np.sqrt(w_g @ cov @ w_g):6.2%}, Sharpe {sr(w_g):.2f}")
print(f"Tangency  : excess mean {w_t @ mu_excess:6.2%}, vol {np.sqrt(w_t @ cov @ w_t):6.2%}, Sharpe {sr(w_t):.2f}")
print(f"Max Sharpe from theory sqrt(mu' S^-1 mu) = {np.sqrt(C):.2f}")
w_eq = np.ones(N) / N
print(f"1/N       : Sharpe {sr(w_eq):.2f}")
# Two-fund separation: any frontier portfolio is a mix of GMV and tangency.
target = 0.10
lam = (target - w_g @ mu_excess) / (w_t @ mu_excess - w_g @ mu_excess)
w_mix = (1 - lam) * w_g + lam * w_t
w_direct = inv @ (((C - B * target) * np.ones(N) + (A * target - B) * mu_excess) / D)
print(f"Two-fund separation check, max |difference| in weights: {np.abs(w_mix - w_direct).max():.1e}")
assert np.allclose(w_mix, w_direct) and abs(sr(w_t) - np.sqrt(C)) < 1e-8


# ---------------------------------------------------------------------------
header("B. Out of sample: does optimisation beat 1/N? (DeMiguel, Garlappi, Uppal 2009)")
# ---------------------------------------------------------------------------
window, hold = TRADING_DAYS, 21
results = {k: [] for k in ["1/N", "GMV sample", "GMV Ledoit-Wolf", "Tangency sample", "Tangency long-only"]}
for seed in range(6):
    mk = simulate_market(n_assets=N, n_days=TRADING_DAYS * 6, seed=200 + seed)
    R = mk.returns.to_numpy() - mk.rf
    oos = {k: [] for k in results}
    for start in range(window, len(R) - hold, hold):
        X = R[start - window : start]
        mu_hat, S = X.mean(axis=0), np.cov(X.T)
        weights = {
            "1/N": np.ones(N) / N,
            "GMV sample": gmv(S),
            "GMV Ledoit-Wolf": gmv(ledoit_wolf(X)),
            "Tangency sample": tangency(mu_hat, S),
        }
        res = minimize(
            lambda w: -(w @ mu_hat) / np.sqrt(w @ S @ w),
            np.ones(N) / N,
            bounds=[(0, 1)] * N,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            method="SLSQP",
        )
        weights["Tangency long-only"] = res.x
        for k, w in weights.items():
            oos[k].append(R[start : start + hold] @ w)
    for k in results:
        r = np.concatenate(oos[k])
        results[k].append(np.sqrt(TRADING_DAYS) * r.mean() / r.std())
table = pd.DataFrame(results).agg(["mean", "std"]).T.round(2)
table.columns = ["OOS Sharpe (mean over paths)", "std"]
print(table.to_string())
print(
    "\nThe unconstrained sample tangency portfolio is the worst: the mean is estimated\n"
    "with an error of order sigma/sqrt(T), larger than the premia themselves.\n"
    "The Backtest notebook's Markowitz baseline (examples/Stock_NeurIPS2018_Backtest.ipynb,\n"
    "EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))) has the same issue."
)
assert table.iloc[:, 0]["Tangency sample"] < table.iloc[:, 0]["1/N"]


# ---------------------------------------------------------------------------
header("C. The portfolio environment's action space")
# ---------------------------------------------------------------------------
# env_portfolio.py:96 sets action_space = Box(low=0, high=1). SB3 clips actions to
# that box, then step() maps them to weights with a softmax (env_portfolio.py:166).
for n in [5, 30, 100]:
    w_max = np.e / (np.e + n - 1)
    w_min = 1 / (1 + (n - 1) * np.e)
    print(f"N = {n:3d}: reachable weights per asset lie in [{w_min:.3f}, {w_max:.3f}]  (1/N = {1/n:.3f})")

# Best Sharpe the agent could ever reach, with the true parameters.
best_softmax = minimize(
    lambda a: -sr(softmax(a)), np.full(N, 0.5), bounds=[(0, 1)] * N, method="L-BFGS-B"
)
long_only = minimize(
    lambda w: -sr(w), np.ones(N) / N, bounds=[(0, 1)] * N,
    constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP",
)
print(f"\nTrue-parameter Sharpe ratios (N = {N}):")
print(f"  1/N                              : {sr(w_eq):.2f}")
print(f"  best portfolio the softmax allows: {-best_softmax.fun:.2f}")
print(f"  best long-only portfolio         : {-long_only.fun:.2f}")
print(f"  unconstrained tangency           : {sr(w_t):.2f}")
print(
    "Even a perfect agent can only tilt each weight by a factor of e around 1/N.\n"
    "Comparing the RL agent with the min-variance or Markowitz baselines is therefore\n"
    "partly a comparison of action spaces, not of learning."
)
assert np.e / (np.e + 29) < 0.09
print("\nAll Module 03 checks passed.")
