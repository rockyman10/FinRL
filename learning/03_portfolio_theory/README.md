# Module 03: Portfolio theory and estimation risk

Markowitz (1952) is the first model every finance PhD learns and the first one
practitioners stop using as written. This module derives mean-variance theory,
shows why plug-in estimates of it perform badly out of sample, and then asks
what FinRL's RL portfolio agent is actually able to choose.

**Prerequisites:** Module 02, constrained optimisation (Lagrangians, KKT).
**Lab:** `python learning/03_portfolio_theory/lab.py`

## Learning objectives

1. Derive the minimum-variance frontier, the global minimum-variance (GMV)
   portfolio, the tangency portfolio and two-fund separation.
2. Show that the maximum Sharpe ratio equals $\sqrt{\mu^\top\Sigma^{-1}\mu}$ and
   connect it to the Hansen-Jagannathan bound.
3. Explain why estimated mean-variance portfolios often lose to 1/N, and which
   fixes (constraints, shrinkage, GMV) help.
4. Judge whether an RL portfolio policy and a Markowitz baseline are being
   compared on equal terms.

## Theory

**Frontier.** Minimise $\tfrac12 w^\top\Sigma w$ subject to $w^\top\mu = m$ and
$w^\top\mathbf 1 = 1$. With $A = \mathbf 1^\top\Sigma^{-1}\mathbf 1$,
$B = \mathbf 1^\top\Sigma^{-1}\mu$, $C = \mu^\top\Sigma^{-1}\mu$ and $D = AC-B^2$,

$$w^*(m) = \Sigma^{-1}\frac{(C - Bm)\mathbf 1 + (Am - B)\mu}{D}, \qquad
\sigma^2(m) = \frac{Am^2 - 2Bm + C}{D}.$$

$w^*(m)$ is linear in $m$, which gives two-fund separation. With a riskless
asset, every investor holds the tangency portfolio
$w_T \propto \Sigma^{-1}(\mu - r_f\mathbf 1)$, and its squared Sharpe ratio is
$(\mu-r_f)^\top\Sigma^{-1}(\mu-r_f)$. That is the largest Sharpe ratio
available from these assets. It is also the Hansen-Jagannathan (1991) lower
bound on the volatility of any stochastic discount factor that prices them
(Module 04).

**Estimation risk.** The error in $\hat\mu$ is about $\sigma/\sqrt{T}$ per
year of data. For a stock with 25% volatility, 10 years of data gives a
standard error of about 8% on the mean, which is larger than the equity
premium. DeMiguel, Garlappi and Uppal (2009) show that none of 14 optimising
models reliably beat 1/N out of sample. They estimate that sample tangency
needs about 3,000 months of data for 25 assets to win. The remedies are
constraints (Jagannathan and Ma 2003 show that a long-only constraint acts as
shrinkage), ignoring means (GMV), shrinking means and covariances, and Bayesian
approaches such as Black-Litterman.

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:96` | `action_space = Box(low=0, high=1)`. |
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:166` and `225-229` | Actions mapped to weights by a softmax. |
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:183-188` | Portfolio return as $\sum_i w_i R_i$ for one day. |
| `examples/Stock_NeurIPS2018_Backtest.ipynb` | The "Mean Variance Optimization" baseline with PyPortfolioOpt `EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))`. |
| `docs/source/tutorial/Introduction/PortfolioAllocation.rst` | The portfolio tutorial, comparing against min-variance. |

## Reading the code critically

- **The softmax box.** Actions in $[0,1]$ passed through a softmax can only
  produce weights between $1/(1+(N-1)e)$ and $e/(e+N-1)$. For the Dow 30 that
  is 1.3% to 8.6%. Every reachable portfolio is a tilt of 1/N, and none of
  them hold zero or short positions. Part C shows the best Sharpe ratio any
  agent can reach in this space with perfect knowledge.
- **Unequal baselines.** The Markowitz baseline estimates $\mu$ from a short
  history, which is the setting where it is known to fail. A fairer benchmark
  set is 1/N, GMV with shrinkage, and long-only tangency.

## Exercises

1. *(Core)* Derive $w^*(m)$ with Lagrange multipliers, and verify the formula
   numerically against the lab.
2. *(Core)* Change the softmax so that actions in $[-k, k]$ are used. How
   large must $k$ be for the softmax space to contain the long-only optimum
   from Part C?
3. *(Advanced)* Add a Black-Litterman portfolio to Part B, with the market
   (equal-weight here) as prior and a single view, and compare.
4. *(Advanced)* Implement the Kan and Zhou (2007) three-fund rule and add it
   to Part B.
5. *(Research)* Replace the softmax with a sparse map (sparsemax, Martins and
   Astudillo 2016) and retrain the portfolio agent from the tutorial. Does
   the agent learn to concentrate?

## Reading

- Markowitz (1952), "Portfolio Selection", *JF*.
- Merton (1972), "An Analytic Derivation of the Efficient Portfolio Frontier", *JFQA*.
- Cochrane (2005), *Asset Pricing*, rev. ed., ch. 5 (mean-variance frontier and the SDF).
- DeMiguel, Garlappi and Uppal (2009), "Optimal Versus Naive Diversification", *RFS*.
- Jagannathan and Ma (2003), "Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps", *JF*.
- Kan and Zhou (2007), "Optimal Portfolio Choice with Parameter Uncertainty", *JFQA*.
- Black and Litterman (1992), "Global Portfolio Optimization", *FAJ*.
