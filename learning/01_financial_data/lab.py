"""Module 01 lab: how data handling choices bias results.

Run:  python learning/01_financial_data/lab.py

Parts
  A. Survivorship bias from FinRL's `clean_data` (drop any ticker with a gap).
  B. Look-ahead bias, and the ffill/bfill step at the end of `preprocess_data`.
  C. Corporate actions: adjusted close next to unadjusted high/low.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import header, sharpe, simulate_market, to_finrl_long_format  # noqa: E402


def finrl_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Faithful copy of FeatureEngineer.clean_data
    (finrl/meta/preprocessor/preprocessors.py:109)."""
    df = df.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    return df[df.tic.isin(tics)]


# ---------------------------------------------------------------------------
header("A. Survivorship bias")
# ---------------------------------------------------------------------------
gaps = []
for seed in range(20):
    m = simulate_market(n_assets=100, seed=seed, delist_below=0.3)
    # The investable universe: an equal-weight portfolio of every stock alive
    # at the start of each day, including the delisting return.
    true_ew = m.returns.mean(axis=1, skipna=True)

    long = to_finrl_long_format(m.returns)
    survivors = finrl_clean_data(long).tic.unique()
    surv_ew = m.returns[survivors].mean(axis=1)

    gaps.append((true_ew.mean() * 252, surv_ew.mean() * 252, len(survivors)))

gaps = pd.DataFrame(gaps, columns=["universe", "survivors_only", "n_survivors"])
print(gaps.describe().loc[["mean", "std"]].round(4))
bias = (gaps.survivors_only - gaps.universe).mean()
print(f"\nAverage survivorship bias in annual mean return: {bias:.2%}")
print(
    "clean_data keeps only tickers with a complete close history, so every stock\n"
    "that was delisted for bad performance is silently removed. FinRL's DOW_30_TICKER\n"
    "list (finrl/config_tickers.py:7) has the same problem: it is *today's* index."
)
assert bias > 0


# ---------------------------------------------------------------------------
header("B1. Look-ahead bias: using information you did not have")
# ---------------------------------------------------------------------------
m = simulate_market(n_assets=1, seed=3)
r = m.returns.iloc[:, 0]
# A rule that goes long after an up day and short after a down day.
signal = np.sign(r)
honest = signal.shift(1) * r  # decide at close t, earn r_{t+1}
cheating = signal * r  # earns the return that produced the signal
print(f"honest Sharpe   : {sharpe(honest):6.2f}  (only regime persistence; one noisy path)")
print(f"look-ahead Sharpe: {sharpe(cheating):6.2f}  <- impossible, it is just |r|")
print(
    "FinRL's environments get the timing right: the agent observes close_t, trades at\n"
    "close_t and is paid with close_{t+1} (env_stocktrading.py:310-351)."
)


# ---------------------------------------------------------------------------
header("B2. The fill step in FeatureEngineer.preprocess_data")
# ---------------------------------------------------------------------------
# preprocessors.py:106 runs df.fillna(method="ffill").fillna(method="bfill") on
# the LONG frame sorted by (date, tic). Rows of different tickers are adjacent,
# so a gap in one stock is filled with another stock's value, and leading NaNs
# of a rolling indicator are back-filled from the future.
toy = pd.DataFrame(
    {
        "date": ["d1", "d1", "d2", "d2", "d3", "d3"],
        "tic": ["AAA", "ZZZ", "AAA", "ZZZ", "AAA", "ZZZ"],
        "sma_2": [np.nan, np.nan, 10.0, 500.0, 11.0, np.nan],
    }
)
filled = toy.ffill().bfill()
print(pd.concat([toy, filled["sma_2"].rename("after_fill")], axis=1).to_string(index=False))
print(
    "\nZZZ's missing d3 value becomes AAA's 11.0 (cross-sectional contamination), and\n"
    "both d1 values become 10.0, which is AAA's d2 value (look-ahead).\n"
    "Correct version: df.groupby('tic').ffill(), and drop the warm-up rows instead of bfill."
)
assert filled.loc[5, "sma_2"] == 11.0 and filled.loc[1, "sma_2"] == 10.0


# ---------------------------------------------------------------------------
header("C. Corporate actions: adjusted vs unadjusted prices")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
n = 60
true_r = rng.normal(0.0005, 0.015, n)
adj_close = 100 * np.cumprod(1 + true_r)
split_day = 30
# Raw prices: before the 2:1 split everything trades at twice the adjusted level.
factor = np.where(np.arange(n) < split_day, 2.0, 1.0)
raw_close = adj_close * factor
raw_high = raw_close * (1 + np.abs(rng.normal(0, 0.005, n)))
raw_low = raw_close * (1 - np.abs(rng.normal(0, 0.005, n)))

raw_ret = pd.Series(raw_close).pct_change()
print(f"raw close return on split day     : {raw_ret[split_day]:+.1%}")
print(f"adjusted close return on split day: {true_r[split_day]:+.1%}")

# YahooDownloader.fetch_data (yahoodownloader.py:69-72) overwrites close with the
# adjusted close but leaves open/high/low unadjusted.
finrl_close = adj_close
outside = np.mean((finrl_close < raw_low) | (finrl_close > raw_high))
print(f"days where FinRL's close lies outside [low, high]: {outside:.0%}")
print(
    "Two default indicators in finrl/config.py:21 (cci_30 and dx_30) mix high, low and\n"
    "close, so they are wrong on every pre-split (and, for dividends, every pre-ex-date)\n"
    "day. Fix: scale open/high/low by adj_close / close before computing indicators."
)
assert outside > 0.4
print("\nAll Module 01 checks passed.")
