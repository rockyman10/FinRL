"""Module 04 lab: factor models, the GRS test, Fama-MacBeth, and benchmark choice.

Run:  python learning/04_asset_pricing/lab.py

Parts
  A. Time-series regressions and the GRS test: CAPM versus the true 3-factor model.
  B. Fama-MacBeth cross-sectional regressions: recovering factor risk premia.
  C. What "alpha" in FinRL's backtest_plot measures (price index benchmark, rf = 0).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import TRADING_DAYS, header, ols, simulate_market  # noqa: E402


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    return (1 + daily).groupby(daily.index.to_period("M")).prod() - 1


def grs(excess: np.ndarray, factors: np.ndarray) -> tuple[float, float]:
    """Gibbons, Ross and Shanken (1989) test that all intercepts are zero."""
    T, N = excess.shape
    L = factors.shape[1]
    X = np.column_stack([np.ones(T), factors])
    coef, *_ = np.linalg.lstsq(X, excess, rcond=None)
    alpha = coef[0]
    resid = excess - X @ coef
    Sigma = resid.T @ resid / (T - L - 1)
    mu_f = factors.mean(axis=0)
    Omega = np.atleast_2d(np.cov(factors.T, ddof=1))
    stat = (T / N) * ((T - N - L) / (T - L - 1)) * (alpha @ np.linalg.solve(Sigma, alpha)) / (
        1 + mu_f @ np.linalg.solve(Omega, mu_f)
    )
    return float(stat), float(1 - stats.f.cdf(stat, N, T - N - L))


# Individual stocks are too noisy to test anything: Fama and French test on 25
# portfolios sorted on size and book-to-market. Here we sort 500 stocks into a
# 5 x 5 grid on their SMB and HML loadings (in practice you would sort on the
# characteristics, which you can observe, not on the true loadings).
# Premia of 7% (MKT), 4% (SMB) and 6% (HML) a year are close to the 1963-1990
# sample in Fama and French (1993).
PREMIA = (0.07, 0.04, 0.06)


def make_portfolios(seed: int, years: int = 30):
    m = simulate_market(
        n_assets=500, n_days=TRADING_DAYS * years, seed=seed, p_calm_to_crisis=0.0, premia=PREMIA, alpha_vol=0.0
    )
    stock_R = to_monthly(m.returns) - ((1 + m.rf) ** 21 - 1)  # monthly excess returns
    groups = pd.qcut(m.betas["SMB"], 5, labels=False) * 5 + pd.qcut(m.betas["HML"], 5, labels=False)
    R = stock_R.T.groupby(groups).mean().T  # 25 equal-weight portfolios
    R.columns = [f"P{c:02d}" for c in R.columns]
    return R, to_monthly(m.factors)


# ---------------------------------------------------------------------------
header("A. GRS tests on 25 portfolios: size and power over 40 simulated histories")
# ---------------------------------------------------------------------------
rows = []
for seed in range(40):
    R, F = make_portfolios(seed=400 + seed)
    for name, cols in [("CAPM (MKT only)", ["MKT"]), ("3-factor (MKT, SMB, HML)", ["MKT", "SMB", "HML"])]:
        stat, p = grs(R.to_numpy(), F[cols].to_numpy())
        rows.append({"model": name, "GRS": stat, "reject at 5%": p < 0.05})
grs_table = pd.DataFrame(rows).groupby("model").mean()
print("30 years of monthly data per history, 25 portfolios from 500 stocks")
print(grs_table.round(2).to_string())
print(
    "\nThe 3-factor model is true, so it should be rejected about 5% of the time (the\n"
    "test's size). With 40 histories that rate has a standard error of about 3.5 points,\n"
    "and compounding daily returns into monthly ones adds a small misspecification.\n"
    "CAPM omits two priced factors; their premia move into the intercepts and GRS\n"
    "rejects much more often (the test's power), but not always. Alpha is always relative\n"
    "to a model, and a non-rejection is weak evidence for it."
)
assert grs_table.loc["CAPM (MKT only)", "reject at 5%"] > 2 * grs_table.loc["3-factor (MKT, SMB, HML)", "reject at 5%"]

R, F = make_portfolios(seed=41)


# ---------------------------------------------------------------------------
header("B. Fama-MacBeth (1973): estimating risk premia from the cross-section")
# ---------------------------------------------------------------------------
first = 60  # 5 years of data to estimate betas, then roll forward one month at a time
lambdas = []
for t in range(first, len(R)):
    win_R, win_F = R.iloc[t - first : t], F.iloc[t - first : t]
    X = np.column_stack([np.ones(first), win_F.to_numpy()])
    B_hat = np.linalg.lstsq(X, win_R.to_numpy(), rcond=None)[0][1:].T  # 25 x 3
    lam, *_ = ols(R.iloc[t].to_numpy(), B_hat)
    lambdas.append(lam)
lambdas = pd.DataFrame(lambdas, columns=["const", "MKT", "SMB", "HML"])
fm = pd.DataFrame(
    {
        "estimate (ann.)": lambdas.mean() * 12,
        "FM s.e. (ann.)": lambdas.std() / np.sqrt(len(lambdas)) * 12,
        "factor sample mean (ann.)": [0.0, *(F.iloc[first:].mean() * 12)],
        "population (ann.)": [0.0, *PREMIA],
    }
)
fm["t-stat"] = fm["estimate (ann.)"] / fm["FM s.e. (ann.)"]
print(fm.round(3).to_string())
print(
    "\nFor a traded factor the premium is its mean, so the cross-sectional estimate\n"
    "should match the factor's sample mean. SMB and HML come out close. The market\n"
    "premium does not: these portfolios were sorted on SMB and HML loadings, so their\n"
    "market betas are all near 1 and the cross-section cannot separate the market\n"
    "premium from the intercept. That is the flat beta-return relation of Fama and\n"
    "French (1992). Betas are estimated, so these standard errors are also too small\n"
    "(Shanken 1992); see exercise 2."
)


# ---------------------------------------------------------------------------
header("C. What 'alpha' means in FinRL's backtest_plot")
# ---------------------------------------------------------------------------
# finrl/plot.py:46-69 benchmarks a strategy built on ADJUSTED closes (total return)
# against ^DJI, a PRICE index without dividends, and pyfolio uses rf = 0.
rng = np.random.default_rng(5)
n = TRADING_DAYS * 10
div_yield, rf = 0.02, 0.03
price_ret = rng.normal(0.06 / TRADING_DAYS, 0.15 / np.sqrt(TRADING_DAYS), n)
total_ret = price_ret + div_yield / TRADING_DAYS
rf_d = rf / TRADING_DAYS
beta_true = 0.6
# A strategy with zero true alpha: 60% index (total return) + 40% cash.
strategy = rf_d + beta_true * (total_ret - rf_d)

cases = {
    "correct: excess returns, total-return index": (strategy - rf_d, total_ret - rf_d),
    "FinRL/pyfolio: rf = 0, price index": (strategy, price_ret),
}
for name, (y, x) in cases.items():
    b, se, _ = ols(y, x)
    print(f"{name:45s} alpha = {b[0] * TRADING_DAYS:+.2%} a year, beta = {b[1]:.2f}")
print(
    "Expected spurious alpha = beta * dividend yield + (1 - beta) * rf\n"
    f"                       = {beta_true * div_yield + (1 - beta_true) * rf:.2%} a year."
)
b, *_ = ols(strategy, price_ret)
assert abs(b[0] * TRADING_DAYS - (beta_true * div_yield + (1 - beta_true) * rf)) < 1e-6
print("\nAll Module 04 checks passed.")
