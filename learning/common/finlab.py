"""Shared helpers for the FinRL learning labs.

Everything here depends only on numpy, pandas and scipy, so the labs run even
when FinRL's own (early-2023 pinned) dependencies are not installed.

The central tool is `simulate_market`, a regime-switching factor model. Using
simulated data with *known* parameters is deliberate: it lets every lab check
an estimator against the truth, which real data never allows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Simulated market
# ---------------------------------------------------------------------------
@dataclass
class Market:
    returns: pd.DataFrame  # simple daily returns, dates x tickers (NaN after delisting)
    factors: pd.DataFrame  # daily factor returns: MKT (excess), SMB, HML
    betas: pd.DataFrame  # true factor loadings, tickers x factors
    alphas: pd.Series  # true daily alphas
    regime: pd.Series  # 0 = calm, 1 = crisis
    rf: float  # daily risk-free rate
    delisted: pd.Series  # delisting date per ticker (NaT if it survives)
    factor_cov_calm: np.ndarray  # true daily factor covariance in the calm regime
    idio_vol_calm: pd.Series  # true daily idiosyncratic volatility in the calm regime

    def true_cov_calm(self) -> np.ndarray:
        """True daily covariance of asset returns in the calm regime."""
        B = self.betas.to_numpy()
        return B @ self.factor_cov_calm @ B.T + np.diag(self.idio_vol_calm.to_numpy() ** 2)


def simulate_market(
    n_assets: int = 30,
    n_days: int = TRADING_DAYS * 8,
    seed: int = 0,
    p_calm_to_crisis: float = 1 / 250,
    p_crisis_to_calm: float = 1 / 40,
    delist_below: float | None = None,
    delisting_return: float = -0.30,
    start: str = "2010-01-04",
    premia: tuple[float, float, float] = (0.07, 0.02, 0.03),
    alpha_vol: float = 0.01,
) -> Market:
    """Simulate daily returns from a 3-factor model with a hidden two-state regime.

    r_it = rf + alpha_i + beta_i' f_t + e_it

    In the crisis regime factor volatilities triple, the market premium turns
    negative and idiosyncratic volatility rises, so correlations go up exactly
    when diversification is needed most (Longin and Solnik 2001).

    If `delist_below` is set (e.g. 0.3), a stock whose cumulative price falls
    below that fraction of its starting price is delisted: it earns
    `delisting_return` on its last day and NaN afterwards (Shumway 1997).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_assets)]
    rf = 0.02 / TRADING_DAYS

    # Hidden regime: two-state Markov chain.
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        p = p_calm_to_crisis if regime[t - 1] == 0 else 1 - p_crisis_to_calm
        regime[t] = rng.random() < p

    # Factor returns (daily). `premia` are the annual means in the calm regime.
    mu_calm = np.array(premia) / TRADING_DAYS
    mu_crisis = np.array([-0.40, -0.05, 0.00]) / TRADING_DAYS
    vol_calm = np.array([0.15, 0.08, 0.08]) / np.sqrt(TRADING_DAYS)
    corr = np.array([[1.0, 0.2, -0.2], [0.2, 1.0, 0.0], [-0.2, 0.0, 1.0]])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, 3)) @ chol.T
    vol = np.where(regime[:, None] == 1, 3 * vol_calm, vol_calm)
    mu = np.where(regime[:, None] == 1, mu_crisis, mu_calm)
    f = mu + vol * z

    betas = np.column_stack(
        [
            rng.uniform(0.6, 1.5, n_assets),
            rng.normal(0.0, 0.5, n_assets),
            rng.normal(0.0, 0.5, n_assets),
        ]
    )
    alphas = rng.normal(0.0, alpha_vol, n_assets) / TRADING_DAYS  # annual cross-sectional sd
    idio = rng.uniform(0.15, 0.35, n_assets) / np.sqrt(TRADING_DAYS)
    idio_t = np.where(regime[:, None] == 1, 1.5 * idio, idio)
    eps = idio_t * rng.standard_normal((n_days, n_assets))

    r = rf + alphas + f @ betas.T + eps

    delisted = pd.Series(pd.NaT, index=tickers, dtype="datetime64[ns]")
    if delist_below is not None:
        level = np.cumprod(1 + r, axis=0)
        for j in range(n_assets):
            hit = np.nonzero(level[:, j] < delist_below)[0]
            if hit.size:
                t = hit[0]
                r[t, j] = delisting_return
                r[t + 1 :, j] = np.nan
                delisted.iloc[j] = dates[t]

    return Market(
        returns=pd.DataFrame(r, index=dates, columns=tickers),
        factors=pd.DataFrame(f, index=dates, columns=["MKT", "SMB", "HML"]),
        betas=pd.DataFrame(betas, index=tickers, columns=["MKT", "SMB", "HML"]),
        alphas=pd.Series(alphas, index=tickers),
        regime=pd.Series(regime, index=dates, name="regime"),
        rf=rf,
        delisted=delisted,
        factor_cov_calm=np.outer(vol_calm, vol_calm) * corr,
        idio_vol_calm=pd.Series(idio, index=tickers),
    )


def to_finrl_long_format(returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
    """Turn a wide return panel into FinRL's long `date, tic, close` format.

    This is the shape FinRL's `FeatureEngineer` and environments consume
    (see finrl/meta/preprocessor/preprocessors.py).
    """
    prices = start_price * (1 + returns).cumprod()
    first = pd.DataFrame(start_price, index=[returns.index[0] - pd.offsets.BDay()], columns=returns.columns)
    prices = pd.concat([first, prices])
    long = prices.stack(future_stack=True).rename("close").reset_index()
    long.columns = ["date", "tic", "close"]
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    return long.sort_values(["date", "tic"], ignore_index=True)


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------
def ann_return(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(pd.Series(r).std() * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    x = pd.Series(r).dropna() - rf
    return float(np.sqrt(periods) * x.mean() / x.std())


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    x = pd.Series(r).dropna() - rf
    downside = np.sqrt((np.minimum(x, 0) ** 2).mean())
    return float(np.sqrt(TRADING_DAYS) * x.mean() / downside)


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(r: pd.Series, rf: float = 0.0) -> pd.Series:
    return pd.Series(
        {
            "ann_return": ann_return(r),
            "ann_vol": ann_vol(r),
            "sharpe": sharpe(r, rf),
            "sortino": sortino(r, rf),
            "max_drawdown": max_drawdown(r),
        }
    )


# ---------------------------------------------------------------------------
# Econometrics
# ---------------------------------------------------------------------------
def ols(y, X, add_const: bool = True, nw_lags: int | None = None):
    """OLS with classical or Newey-West (1987) standard errors.

    Returns (coefficients, standard errors, residuals).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    if nw_lags is None:
        cov = XtX_inv * (e @ e) / (n - k)
    else:
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for lag in range(1, nw_lags + 1):
            w = 1 - lag / (nw_lags + 1)  # Bartlett kernel
            G = Xe[lag:].T @ Xe[:-lag]
            S += w * (G + G.T)
        cov = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(cov)), e


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
