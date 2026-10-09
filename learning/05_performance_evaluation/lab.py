"""Module 05 lab: how much a Sharpe ratio tells you, and how selection inflates it.

Run:  python learning/05_performance_evaluation/lab.py

Parts
  A. The standard error of a Sharpe ratio (Lo 2002; Mertens 2002).
  B. Multiple testing: the best of many zero-skill strategies, and the deflated Sharpe ratio.
  C. The ensemble agent's model selection on 63-day validation Sharpe ratios.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import TRADING_DAYS, header  # noqa: E402

rng = np.random.default_rng(55)
EULER_GAMMA = 0.5772156649


def sharpe_se(r: np.ndarray) -> float:
    """Per-period Sharpe standard error allowing skewness and kurtosis (Mertens 2002)."""
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    return float(np.sqrt((1 + 0.5 * sr**2 - g3 * sr + (g4 - 3) / 4 * sr**2) / len(r)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials zero-mean Sharpe estimates (Bailey and Lopez de Prado 2014)."""
    z = stats.norm.ppf
    return float(
        np.sqrt(var_sr)
        * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * np.e)))
    )


def deflated_sharpe(r: np.ndarray, sr0: float) -> float:
    """Probability that the true per-period Sharpe exceeds sr0 (PSR evaluated at sr0)."""
    sr = r.mean() / r.std(ddof=1)
    return float(stats.norm.cdf((sr - sr0) / sharpe_se(r)))


# ---------------------------------------------------------------------------
header("A. How precisely can you measure a Sharpe ratio?")
# ---------------------------------------------------------------------------
true_sr_annual = 0.5
for years in [1, 3, 10, 30]:
    T = TRADING_DAYS * years
    se_annual = np.sqrt((1 + 0.5 * (true_sr_annual / np.sqrt(TRADING_DAYS)) ** 2) / T) * np.sqrt(TRADING_DAYS)
    print(f"{years:2d} years of daily data: s.e. of annualised Sharpe = {se_annual:.2f}")
print("Rule of thumb: s.e.(annual SR) ~ 1 / sqrt(years). A true 0.5 needs ~16 years for t = 2.")

# Coverage of the 95% interval for a short-volatility style strategy: monthly
# returns of +1.2% most of the time and a -12% crash with probability 3%.
def draw(size):
    crash = rng.random(size) < 0.03
    return np.where(crash, -0.12 + 0.03 * rng.standard_normal(size), 0.012 + 0.005 * rng.standard_normal(size))


big = draw(5_000_000)
true_sr = big.mean() / big.std()
T, n_sims, cov_iid, cov_mertens = 120, 4000, 0, 0
for _ in range(n_sims):
    x = draw(T)
    sr_hat = x.mean() / x.std(ddof=1)
    cov_iid += abs(sr_hat - true_sr) < 1.96 * np.sqrt((1 + 0.5 * sr_hat**2) / T)
    cov_mertens += abs(sr_hat - true_sr) < 1.96 * sharpe_se(x)
print(
    f"\nShort-volatility strategy, 10 years of monthly data (annual SR {true_sr * np.sqrt(12):.2f}, "
    f"skew {stats.skew(big):.1f}, kurtosis {stats.kurtosis(big, fisher=False):.0f}):"
)
print(f"  coverage of a nominal 95% interval, normal-theory s.e.: {cov_iid / n_sims:.1%}")
print(f"  coverage with the Mertens (2002) s.e.                 : {cov_mertens / n_sims:.1%}")
print("  Negative skew makes the Sharpe ratio look more precise than it is.")
assert cov_iid / n_sims < 0.8 < cov_mertens / n_sims


# ---------------------------------------------------------------------------
header("B. Multiple testing: 200 strategies with zero skill")
# ---------------------------------------------------------------------------
n_strats, T_is, T_oos = 200, TRADING_DAYS * 3, TRADING_DAYS * 3
R = rng.normal(0, 0.01, (T_is + T_oos, n_strats))  # no strategy has any edge
is_sr = R[:T_is].mean(0) / R[:T_is].std(0) * np.sqrt(TRADING_DAYS)
oos_sr = R[T_is:].mean(0) / R[T_is:].std(0) * np.sqrt(TRADING_DAYS)
best = int(np.argmax(is_sr))
var_sr = np.var(is_sr / np.sqrt(TRADING_DAYS), ddof=1)
sr0 = expected_max_sharpe(n_strats, var_sr)
print(f"best in-sample annual Sharpe : {is_sr[best]:.2f}")
print(f"expected max under the null  : {sr0 * np.sqrt(TRADING_DAYS):.2f}")
print(f"that strategy out of sample  : {oos_sr[best]:.2f}")
psr0 = deflated_sharpe(R[:T_is, best], 0.0)
dsr = deflated_sharpe(R[:T_is, best], sr0)
print(f"P(true SR > 0), ignoring the search  (PSR): {psr0:.1%}")
print(f"P(true SR > 0), accounting for 200 trials (DSR): {dsr:.1%}")
assert psr0 > 0.95 and dsr < 0.9


# ---------------------------------------------------------------------------
header("C. Picking the ensemble's model on a 63-day validation Sharpe")
# ---------------------------------------------------------------------------
# DRLEnsembleAgent (finrl/agents/stablebaselines3/models.py:327-700) trains A2C, PPO
# and DDPG each quarter, computes a Sharpe ratio on the last 63 trading days
# (get_validation_sharpe, models.py:214-230) and trades the next 63 days with the winner.
se_63 = np.sqrt(TRADING_DAYS / 63)
print(f"s.e. of an annualised Sharpe measured over 63 days: about {se_63:.1f}")
print("  ... while the gap between decent strategies is maybe 0.2-0.5.")

skills = np.array([0.3, 0.5, 0.7])  # three models with different true annual Sharpe
n_quarters, picks = 4000, []
sd = 0.15 / np.sqrt(TRADING_DAYS)  # 15% annual volatility
mu_d = skills / np.sqrt(TRADING_DAYS) * sd  # daily means implied by the annual Sharpe ratios
for _ in range(n_quarters):
    val = rng.normal(mu_d, sd, (63, 3))
    val_sr = val.mean(0) / val.std(0) * np.sqrt(4)  # FinRL's annualisation (sqrt(4))
    k = int(np.argmax(val_sr))
    picks.append(k)
picks = np.bincount(picks, minlength=3) / n_quarters
print(pd.Series(picks, index=[f"true SR {s}" for s in skills], name="share of quarters picked").round(2).to_string())
print(f"ensemble true Sharpe ~ {np.dot(picks, skills):.2f}  vs always using the best model: {skills.max():.2f}")
print(
    "Two further details in get_validation_sharpe: it annualises daily returns with\n"
    "sqrt(4) instead of sqrt(252) (harmless for ranking, wrong as a number), and it returns\n"
    "+inf for a model that never traded but had positive mean return."
)
assert picks[2] < 0.6
print("\nAll Module 05 checks passed.")
