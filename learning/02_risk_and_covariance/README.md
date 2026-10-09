# Module 02: Returns, risk and covariance estimation

Almost every quantity in portfolio choice is a function of a covariance
matrix, and the covariance matrix is the thing we estimate worst when the
number of assets is large relative to the sample. This module covers that
problem through FinRL's turbulence index and the covariance state of its
portfolio environment.

**Prerequisites:** Module 01, linear algebra (eigen-decomposition), multivariate normal.
**Lab:** `python learning/02_risk_and_covariance/lab.py`

## Learning objectives

1. Derive the distribution of a Mahalanobis distance under a known and an
   estimated covariance (chi-square versus Hotelling $T^2$/$F$).
2. Interpret the Kritzman-Li turbulence index as an outlier test, and measure
   its power to detect regimes.
3. Explain why the sample covariance is ill-conditioned when $N/T$ is not
   small, and derive the Ledoit-Wolf shrinkage estimator.
4. Evaluate covariance estimators by the out-of-sample risk of the portfolios
   they produce, not by in-sample fit.

## Theory

**Turbulence as a Mahalanobis distance.** Kritzman and Li (2010) define

$$d_t = (r_t - \hat\mu)^\top \hat\Sigma^{-1} (r_t - \hat\mu),$$

with $\hat\mu, \hat\Sigma$ estimated over a trailing window. If returns are
i.i.d. $N(\mu, \Sigma)$ and the parameters are known, $d_t \sim \chi^2_N$.
With both estimated from $T$ past observations,

$$\frac{T(T-N)}{(T-1)(T+1)N}\, d_t \sim F_{N,\,T-N}, \qquad
E[d_t] = \frac{N(T^2-1)}{T(T-N-2)}.$$

For $N=30$ and $T=252$ that is about 34.4, not 30. A threshold that ignores
estimation error flags too many days. The index is large either when returns
are extreme or when they break the usual correlation pattern, which is why it
picks up crisis regimes better than volatility alone.

**Why the sample covariance fails.** The eigenvalues of the sample covariance
are more dispersed than the true ones. In the limit $N/T \to c$ they follow the
Marchenko-Pastur law even when $\Sigma = I$. Mean-variance optimisation inverts
$\hat\Sigma$, so it loads on the smallest, most underestimated eigenvalues.
Michaud (1989) called this "error maximisation".

**Shrinkage.** Ledoit and Wolf (2004) choose
$\hat\Sigma_{LW} = \delta F + (1-\delta) S$ with target $F = mI$ and the
$\delta$ that minimises expected Frobenius loss. All the inputs to that
formula can be estimated consistently. Better targets include the
constant-correlation matrix (Ledoit and Wolf 2003) and a factor-model
covariance (see Module 04). Nonlinear shrinkage (Ledoit and Wolf 2017) shrinks
each eigenvalue separately.

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/meta/preprocessor/preprocessors.py:215-267` | `calculate_turbulence`: 252-day window, mean-centred, `np.linalg.pinv` of the sample covariance, first two positive values zeroed. |
| `finrl/meta/data_processors/processor_yahoofinance.py:282-330` | The same computation in the newer data processor. |
| `finrl/agents/stablebaselines3/models.py:349-416` | The ensemble's in-sample turbulence threshold. The if/else at 401-412 is **dead code**: line 414 always overwrites it with the 0.99 quantile. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:307-309` | When turbulence exceeds the threshold, the agent is forced to sell everything. |
| `docs/source/tutorial/Introduction/PortfolioAllocation.rst:206-216` | How `cov_list` (a 252-day sample covariance per date) is built for the portfolio environment. |
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:98-112` | The covariance matrix *is* the agent's state: an $N \times N$ block stacked with indicators. |

## Reading the code critically

- `pinv` silently copes with a singular $\hat\Sigma$ (for example $N > T$),
  but then the distance is computed only in the estimated column space. A
  shrunk covariance is the principled fix.
- The 0.99 in-sample quantile flags about 1% of days. In the lab its recall
  for the crisis regime is around 10%: the "turbulence liquidation" rule is
  almost never triggered in the regime it was meant for.
- Giving an RL agent a raw sample covariance as state means $N(N+1)/2$ noisy
  inputs. For the Dow 30 that is 465 numbers re-estimated every day from 252
  observations.

## Exercises

1. *(Core)* Replace the 252-day sample covariance in the turbulence index with
   (a) Ledoit-Wolf and (b) an EWMA covariance with $\lambda = 0.94$
   (RiskMetrics). Compare AUC for the crisis regime in the lab.
2. *(Core)* Verify the $F$-distribution result by simulation: draw i.i.d.
   normals and compare the empirical CDF of the scaled $d_t$ with
   `scipy.stats.f`.
3. *(Advanced)* Build a factor-model covariance $\hat B \hat\Sigma_f \hat B^\top + \hat D$
   from the simulated factors and add it to Part B. How close does it get to
   the oracle, and why?
4. *(Advanced)* Implement the Kritzman, Li, Page and Rigobon (2011)
   absorption ratio (share of variance explained by the top eigenvectors) and
   compare it with turbulence as an early-warning signal.
5. *(Research)* Plot the eigenvalue spectrum of a 252-day Dow 30 covariance
   against the Marchenko-Pastur density. How many eigenvalues carry signal?
   Apply eigenvalue clipping (Laloux et al. 2000) and repeat Part B.

## Reading

- Kritzman and Li (2010), "Skulls, Financial Turbulence, and Risk Management", *FAJ*.
- Ledoit and Wolf (2004), "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices", *JMVA*.
- Ledoit and Wolf (2003), "Improved Estimation of the Covariance Matrix of Stock Returns", *JEF*.
- Michaud (1989), "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?", *FAJ*.
- Laloux, Cizeau, Potters and Bouchaud (2000), "Random Matrix Theory and Financial Correlations", *IJTAF*.
- Anderson (2003), *An Introduction to Multivariate Statistical Analysis*, ch. 5 (Hotelling's $T^2$).
