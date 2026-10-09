"""Module 07 lab: from Merton's problem to the MDP inside FinRL.

Run:  python learning/07_dynamic_portfolio_choice/lab.py

Parts
  A. Merton's constant-opportunity-set solution, checked against exact CRRA expected utility.
  B. Reward design: what FinRL's reward (change in wealth) asks the agent to maximise.
  C. Predictable returns: hedging demand by backward induction (Bellman equation).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import header  # noqa: E402

rng = np.random.default_rng(77)
MU, SIGMA, RF = 0.06, 0.16, 0.02  # annual excess return, volatility, risk-free rate


# ---------------------------------------------------------------------------
header("A. Merton (1969): the optimal constant risky share")
# ---------------------------------------------------------------------------
# With continuous rebalancing, log W_T ~ N((r + pi*mu - pi^2 sigma^2 / 2) T, pi^2 sigma^2 T),
# so E[W_T^(1-g)] has a closed form and we can maximise it on a grid.
def crra_certainty_equivalent(pi: np.ndarray, gamma: float, T: float = 10.0) -> np.ndarray:
    m = (RF + pi * MU - 0.5 * pi**2 * SIGMA**2) * T
    v = pi**2 * SIGMA**2 * T
    if gamma == 1:
        return np.exp(m)
    return np.exp(m + 0.5 * (1 - gamma) * v)  # = (E[W^(1-g)])^(1/(1-g))


grid = np.linspace(0, 4, 4001)
for gamma in [1, 2, 5, 10]:
    best = grid[np.argmax(crra_certainty_equivalent(grid, gamma))]
    print(f"gamma = {gamma:2d}: numerical optimum {best:.3f}, Merton mu/(gamma sigma^2) = {MU / (gamma * SIGMA**2):.3f}")
    assert abs(best - MU / (gamma * SIGMA**2)) < 2e-3 or best == grid[-1]
print("The horizon T drops out: with i.i.d. returns and CRRA utility the investor is myopic.")


# ---------------------------------------------------------------------------
header("B. What does FinRL's reward ask for?")
# ---------------------------------------------------------------------------
# StockTradingEnv: reward = (end_total_asset - begin_total_asset) * reward_scaling
# (env_stocktrading.py:349-351). Summed over an episode, that is W_T - W_0: the agent
# is risk neutral in dollars. The best constant policy then uses the most leverage allowed.
years, n_paths, max_lev = 10, 20000, 3.0
policies = {
    "risk neutral (FinRL reward)": max_lev,
    "log utility / Kelly (reward = change in log W)": MU / SIGMA**2,
    "CRRA gamma = 5": MU / (5 * SIGMA**2),
}
rows = []
for name, pi in policies.items():
    z = rng.standard_normal((n_paths, years))  # annual steps are enough for this comparison
    log_w = ((RF + pi * MU - 0.5 * pi**2 * SIGMA**2) + pi * SIGMA * z).sum(axis=1)
    W = np.exp(log_w)
    rows.append(
        {
            "policy": name,
            "risky share": pi,
            "mean W_T": W.mean(),
            "median W_T": np.median(W),
            "P(W_T < 0.5)": (W < 0.5).mean(),
        }
    )
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(
    "The risk-neutral agent has the highest mean and a lower median than Kelly: a few\n"
    "lucky paths carry the mean. Reward scaling (reward_scaling = 1e-4 in the notebooks)\n"
    "changes nothing here; it multiplies the objective by a constant. Only a concave\n"
    "transformation of wealth, such as log returns, changes the risk attitude."
)
assert rows[0]["median W_T"] < rows[1]["median W_T"] and rows[0]["mean W_T"] > rows[1]["mean W_T"]


# ---------------------------------------------------------------------------
header("C. Predictable returns: myopic versus long-horizon allocation")
# ---------------------------------------------------------------------------
# Monthly model in the spirit of Barberis (2000) and Campbell and Viceira (1999):
#   r^e_{t+1} = a + b x_t + e_{t+1},  x_{t+1} = phi x_t + sqrt(1 - phi^2) eta_{t+1},
#   corr(e, eta) = rho < 0 (a fall in prices raises the predictor, like the dividend yield).
a, b, sig_e = 0.005, 0.003, 0.045
phi, rho, rf_m, gamma = 0.98, -0.9, 0.003, 5.0
x_grid = np.linspace(-3, 3, 61)
pi_grid = np.linspace(0, 3, 301)
nodes, weights = np.polynomial.hermite_e.hermegauss(9)  # probabilists' Gauss-Hermite
weights = weights / weights.sum()
Z1, Z2 = np.meshgrid(nodes, nodes, indexing="ij")
W2 = np.outer(weights, weights)
e = sig_e * Z1
eta = rho * Z1 + np.sqrt(1 - rho**2) * Z2


def bellman_step(psi_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """psi_t(x) = min_pi E[R(pi)^(1-g) psi_{t+1}(x')] for gamma > 1 (V = W^(1-g) psi / (1-g))."""
    psi = np.empty_like(x_grid)
    pol = np.empty_like(x_grid)
    for i, x in enumerate(x_grid):
        re = a + b * x + e  # excess return at each quadrature node
        x_next = phi * x + np.sqrt(1 - phi**2) * eta
        psi_n = np.interp(x_next, x_grid, psi_next)
        gross = 1 + rf_m + pi_grid[:, None, None] * re[None]
        obj = (W2[None] * gross ** (1 - gamma) * psi_n[None]).sum(axis=(1, 2))
        k = int(np.argmin(obj))
        psi[i], pol[i] = obj[k], pi_grid[k]
    return psi, pol


psi = np.ones_like(x_grid)
policy_by_horizon = {}
for h in range(1, 241):
    psi, pol = bellman_step(psi)
    if h in (1, 12, 60, 120, 240):
        policy_by_horizon[h] = pol.copy()
table = pd.DataFrame(policy_by_horizon, index=np.round(x_grid, 1)).loc[[-2.0, -1.0, 0.0, 1.0, 2.0]]
table.columns = [f"{h} months" for h in table.columns]
table.index.name = "predictor x (sd units)"
print("Optimal risky share by investment horizon:")
print(table.round(2).to_string())
myopic = (a + b * x_grid) / (gamma * sig_e**2)
print(f"\nmyopic formula at x = 0: {myopic[30]:.2f}")
print(
    "Long-horizon investors hold more equity than the myopic rule: with rho < 0, bad\n"
    "returns raise future expected returns, so stocks hedge their own reinvestment risk\n"
    "(intertemporal hedging demand, Merton 1973). An RL agent with a state that includes\n"
    "x and a utility-based reward is trying to learn exactly this policy (Module 08)."
)
assert policy_by_horizon[120][30] > policy_by_horizon[1][30] + 0.1
print("\nAll Module 07 checks passed.")
