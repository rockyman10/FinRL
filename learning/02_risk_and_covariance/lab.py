"""Module 02 lab: covariance estimation and the turbulence index.

Run:  python learning/02_risk_and_covariance/lab.py

Parts
  A. FinRL's turbulence index: its null distribution and its power to detect crises.
  B. Sample covariance versus Ledoit-Wolf shrinkage when N/T is not small.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import header, simulate_market, to_finrl_long_format  # noqa: E402


def finrl_calculate_turbulence(data: pd.DataFrame) -> pd.DataFrame:
    """Faithful copy of FeatureEngineer.calculate_turbulence
    (finrl/meta/preprocessor/preprocessors.py:215-267)."""
    df = data.copy()
    df_price_pivot = df.pivot(index="date", columns="tic", values="close")
    df_price_pivot = df_price_pivot.pct_change()
    unique_date = df.date.unique()
    start = 252
    turbulence_index = [0] * start
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[
            (df_price_pivot.index < unique_date[i])
            & (df_price_pivot.index >= unique_date[i - 252])
        ]
        filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min() :].dropna(axis=1)
        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
            filtered_hist_price, axis=0
        )
        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            turbulence_temp = temp[0][0] if count > 2 else 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)
    return pd.DataFrame({"date": df_price_pivot.index, "turbulence": turbulence_index})


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic."""
    ranks = pd.Series(score).rank().to_numpy()
    n1 = label.sum()
    n0 = len(label) - n1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit and Wolf (2004, JMVA) shrinkage towards a scaled identity."""
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    m = np.trace(S) / N
    d2 = np.linalg.norm(S - m * np.eye(N), "fro") ** 2 / N
    b2_bar = sum(np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in Xc) / T**2 / N
    b2 = min(b2_bar, d2)
    delta = b2 / d2
    return delta * m * np.eye(N) + (1 - delta) * S, delta


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    ones = np.ones(len(cov))
    w = np.linalg.solve(cov, ones)
    return w / w.sum()


# ---------------------------------------------------------------------------
header("A. The turbulence index (Kritzman and Li 2010)")
# ---------------------------------------------------------------------------
N, T = 30, 252
m = simulate_market(n_assets=N, n_days=T * 10, seed=11)
long = to_finrl_long_format(m.returns)
turb = finrl_calculate_turbulence(long).set_index("date")["turbulence"]
turb.index = pd.to_datetime(turb.index)
turb = turb.iloc[1:]  # first row is the artificial starting price
turb = turb.loc[m.returns.index]
regime = m.regime.loc[turb.index]
valid = turb.index >= turb.index[T + 3]
turb, regime = turb[valid], regime[valid]

expected_null = N * (T**2 - 1) / (T * (T - N - 2))
print(f"Theory: with an estimated mean and covariance, E[d^2] = N(T^2-1)/(T(T-N-2)) = {expected_null:.1f}")
print(f"Mean turbulence on calm days  : {turb[regime == 0].mean():8.1f}"
      "   (a bit lower: windows that contain crisis days inflate the covariance)")
print(f"Mean turbulence on crisis days: {turb[regime == 1].mean():8.1f}")
print(f"AUC of turbulence for the hidden crisis state: {auc(turb.to_numpy(), regime.to_numpy()):.2f}")

q90, q99 = np.quantile(turb, [0.90, 0.99])
for name, q in [("0.90 quantile", q90), ("0.99 quantile", q99)]:
    flag = turb > q
    print(
        f"  threshold = in-sample {name} ({q:6.1f}): flags {flag.mean():5.1%} of days, "
        f"precision {regime[flag].mean():5.1%}, recall {flag[regime == 1].mean():5.1%}"
    )
print(
    "DRLEnsembleAgent computes an if/else between the 0.90 and 1.00 quantiles\n"
    "(models.py:401-412) and then overwrites it with the 0.99 quantile (models.py:414).\n"
    "Only the 0.99 line has any effect."
)
assert turb[regime == 1].mean() > 2 * turb[regime == 0].mean()


# ---------------------------------------------------------------------------
header("B. Sample covariance versus Ledoit-Wolf shrinkage")
# ---------------------------------------------------------------------------
rows = []
for N in [30, 100, 200]:
    for seed in range(10):
        mk = simulate_market(n_assets=N, n_days=T, seed=100 + seed, p_calm_to_crisis=0.0)
        true_cov = mk.true_cov_calm()
        X = mk.returns.to_numpy()
        S = np.cov(X.T)
        LW, delta = ledoit_wolf(X)
        for name, C in [("sample", S), ("ledoit_wolf", LW)]:
            w = gmv_weights(C)
            rows.append(
                {
                    "N": N,
                    "estimator": name,
                    "shrinkage": delta if name == "ledoit_wolf" else 0.0,
                    "cond_number": np.linalg.cond(C),
                    "true_gmv_vol": np.sqrt(w @ true_cov @ w * 252),
                }
            )
        w_star = gmv_weights(true_cov)
        rows.append({"N": N, "estimator": "oracle (true cov)", "shrinkage": np.nan,
                     "cond_number": np.linalg.cond(true_cov),
                     "true_gmv_vol": np.sqrt(w_star @ true_cov @ w_star * 252)})
res = pd.DataFrame(rows).groupby(["N", "estimator"]).mean()
print(res.round(3).to_string())
print(
    "\nWith T = 252 days and N = 200 assets the sample covariance is nearly singular;\n"
    "its GMV portfolio is badly mis-estimated. Shrinkage trades a little bias for a\n"
    "large variance reduction. env_portfolio.py feeds the raw 252-day sample covariance\n"
    "into the agent's state (`cov_list`, built in docs/source/tutorial/Introduction/PortfolioAllocation.rst)."
)
lw = res.xs("ledoit_wolf", level="estimator")["true_gmv_vol"]
sm = res.xs("sample", level="estimator")["true_gmv_vol"]
assert (lw.loc[[100, 200]] < sm.loc[[100, 200]]).all()
print("\nAll Module 02 checks passed.")
