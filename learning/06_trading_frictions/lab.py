"""Module 06 lab: transaction costs, turnover and optimal execution.

Run:  python learning/06_trading_frictions/lab.py

Parts
  A. The portfolio environment accepts a transaction cost and never charges it.
  B. Gross versus net: break-even costs and partial rebalancing (Garleanu and Pedersen 2013).
  C. Optimal liquidation (Almgren and Chriss 2000).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import TRADING_DAYS, header, sharpe, simulate_market  # noqa: E402

rng = np.random.default_rng(66)


def run_weights(R: np.ndarray, W: np.ndarray, cost: float) -> tuple[np.ndarray, float]:
    """Hold target weights W[t] over day t+1, paying `cost` per unit of turnover.

    Turnover is measured against the weights after the previous day's drift, which
    is what an actual rebalance costs.
    """
    net, turnover = [], []
    w_prev = W[0]
    for t in range(len(R) - 1):
        trade = np.abs(W[t] - w_prev).sum()
        gross = W[t] @ R[t + 1]
        net.append(gross - cost * trade)
        turnover.append(trade)
        drifted = W[t] * (1 + R[t + 1])
        w_prev = drifted / drifted.sum()
    return np.array(net), float(np.mean(turnover))


# ---------------------------------------------------------------------------
header("A. StockPortfolioEnv's unused transaction_cost_pct")
# ---------------------------------------------------------------------------
# env_portfolio.py:89 stores self.transaction_cost_pct. Nothing else reads it: step()
# computes the portfolio return as sum(w * R) (env_portfolio.py:183-188) with no cost.
N = 30
m = simulate_market(n_assets=N, n_days=TRADING_DAYS * 5, seed=61)
R = m.returns.to_numpy()
equal = np.full((len(R), N), 1 / N)
# A stand-in for an RL policy: softmax of fresh actions in [0, 1] every day.
noisy = np.exp(rng.uniform(0, 1, (len(R), N)))
noisy /= noisy.sum(axis=1, keepdims=True)
rows = []
for name, W in [("1/N, rebalanced daily", equal), ("softmax policy, new actions daily", noisy)]:
    for cost in [0.0, 0.001]:
        net, to = run_weights(R, W, cost)
        rows.append({"policy": name, "cost per $ traded": cost, "daily turnover": to,
                     "annual cost drag": to * cost * TRADING_DAYS, "Sharpe": sharpe(net)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(
    "At 10 bp the noisy policy pays several percent a year that the environment never\n"
    "charges, so the agent is trained and evaluated as if trading were free."
)


# ---------------------------------------------------------------------------
header("B. A predictable signal, its break-even cost, and partial rebalancing")
# ---------------------------------------------------------------------------
# One asset whose expected return is a persistent AR(1) signal s_t.
T = TRADING_DAYS * 20
phi, sig_s = 0.97, 0.0005
s = np.zeros(T)
for t in range(1, T):
    s[t] = phi * s[t - 1] + sig_s * np.sqrt(1 - phi**2) * rng.standard_normal()
r_next = s + 0.01 * rng.standard_normal(T)  # realised return over t -> t+1, E = s_t
target = s / (0.01**2) / 50  # mean-variance position with risk aversion 50, in units of wealth


def backtest_position(pos: np.ndarray, cost: float) -> np.ndarray:
    trades = np.abs(np.diff(np.r_[0.0, pos]))
    return pos * r_next - cost * trades


gross = backtest_position(target, 0.0)
turnover = np.abs(np.diff(target)).mean()
breakeven = gross.mean() / turnover
print(f"full rebalancing: gross Sharpe {sharpe(gross):.2f}, daily turnover {turnover:.3f}")
print(f"break-even cost (gross mean / turnover): {breakeven * 1e4:.1f} bp per $ traded")

print("\nNet Sharpe by trading speed (fraction of the gap to target closed each day):")
table = {}
for cost_bp in [0, 2, 5, 10]:
    row = {}
    for speed in [1.0, 0.5, 0.2, 0.1, 0.05]:
        pos = np.zeros(T)
        for t in range(T):
            prev = pos[t - 1] if t else 0.0
            pos[t] = prev + speed * (target[t] - prev)
        row[speed] = sharpe(backtest_position(pos, cost_bp / 1e4))
    table[f"{cost_bp} bp"] = row
table = pd.DataFrame(table).T
table.columns = [f"speed {c}" for c in table.columns]
print(table.round(2).to_string())
print(
    "With costs, trading only part of the way to the target ('aim in front of the\n"
    "target and trade partially towards it') dominates full rebalancing. FinRL's\n"
    "stock environment caps the trade size with hmax but has no such cost-aware policy."
)
assert table.loc["10 bp"].idxmax() != "speed 1.0"


# ---------------------------------------------------------------------------
header("C. Almgren-Chriss optimal liquidation")
# ---------------------------------------------------------------------------
X, n_steps = 1_000_000, 10  # shares to sell over 10 periods (e.g. one day in 10 slices)
sigma = 0.95  # price volatility per period in $ per share (sqrt of variance per period)
eta = 2.5e-6  # temporary impact: price concession of eta * (shares traded per period)
gamma_perm = 2.5e-7  # permanent impact per share


def ac_schedule(lam: float) -> np.ndarray:
    tau = 1.0
    eta_tilde = eta - 0.5 * gamma_perm * tau
    kappa_tilde2 = lam * sigma**2 / eta_tilde
    kappa = np.arccosh(0.5 * kappa_tilde2 * tau**2 + 1) / tau if lam > 0 else 0.0
    t = np.arange(n_steps + 1) * tau
    if kappa == 0:
        return X * (1 - t / (n_steps * tau))
    return X * np.sinh(kappa * (n_steps * tau - t)) / np.sinh(kappa * n_steps * tau)


def cost_and_risk(holdings: np.ndarray) -> tuple[float, float]:
    trades = -np.diff(holdings)
    exp_cost = 0.5 * gamma_perm * X**2 + (eta - 0.5 * gamma_perm) * np.sum(trades**2)
    var = sigma**2 * np.sum(holdings[1:] ** 2)
    return exp_cost, np.sqrt(var)


print(f"{'risk aversion':>14s} {'E[cost] $':>12s} {'sd(cost) $':>12s}   holdings path (% of X)")
for lam in [0.0, 1e-7, 1e-6, 1e-5]:
    h = ac_schedule(lam)
    c, sd = cost_and_risk(h)
    path = " ".join(f"{v:4.0%}" for v in h / X)
    print(f"{lam:14.0e} {c:12,.0f} {sd:12,.0f}   {path}")
print(
    "lambda = 0 gives TWAP (straight line): lowest expected cost, highest risk.\n"
    "Higher risk aversion front-loads the selling, paying more impact to cut variance."
)
c0, s0 = cost_and_risk(ac_schedule(0.0))
c1, s1 = cost_and_risk(ac_schedule(1e-5))
assert c1 > c0 and s1 < s0
print("\nAll Module 06 checks passed.")
