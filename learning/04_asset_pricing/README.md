# Module 04: Asset pricing and factor models

"Alpha" only means something relative to a model of expected returns. This
module builds the stochastic discount factor view of asset pricing, then the
empirical machinery used to test it: time-series regressions, the GRS test and
Fama-MacBeth. It ends with what the alpha and beta in FinRL's backtest report
actually measure.

**Prerequisites:** Modules 02-03, OLS and GMM basics.
**Lab:** `python learning/04_asset_pricing/lab.py`

## Learning objectives

1. Derive the pricing equation $E[m R^e] = 0$, the beta representation, and
   the Hansen-Jagannathan bound.
2. Run and interpret time-series factor regressions and the GRS test,
   including their size and power.
3. Estimate factor risk premia with Fama-MacBeth, and explain the
   errors-in-variables problem and the Shanken correction.
4. Check whether a reported alpha is net of the right benchmark and the
   risk-free rate.

## Theory

**SDF and beta pricing.** No arbitrage implies a positive SDF $m$ with
$E[mR^e]=0$ for every excess return. If $m = a - b^\top f$ is linear in
factors, expected returns take the beta form

$$E[R^e_i] = \beta_i^\top \lambda, \qquad \beta_i = \text{Cov}(f,f)^{-1}\text{Cov}(f, R^e_i).$$

From $E[mR^e]=0$ and Cauchy-Schwarz, $\sigma(m)/E(m) \ge |E[R^e]|/\sigma(R^e)$
for every asset. That is the Hansen-Jagannathan bound, and it links Module 03's
maximum Sharpe ratio to the volatility of marginal utility.

**Time-series tests (traded factors).** Regress
$R^e_{i,t} = \alpha_i + \beta_i^\top f_t + \varepsilon_{i,t}$. The model says
$\alpha = 0$. Gibbons, Ross and Shanken (1989) give the exact finite-sample test

$$\text{GRS} = \frac{T}{N}\,\frac{T-N-L}{T-L-1}\,
\frac{\hat\alpha^\top\hat\Sigma^{-1}\hat\alpha}{1+\bar f^\top\hat\Omega^{-1}\bar f}
\sim F_{N,\,T-N-L}.$$

The numerator is the improvement in squared Sharpe ratio from adding the test
assets to the factors. GRS asks whether the factors span the tangency portfolio.

**Cross-sectional tests and Fama-MacBeth.** Estimate betas, then for each
period regress returns on betas: $R^e_{i,t} = \gamma_{0,t} + \hat\beta_i^\top\lambda_t + u_{i,t}$.
The estimate is the time-series mean of $\hat\lambda_t$, and its standard error
comes from the time-series variance of $\hat\lambda_t$. This handles
cross-sectional correlation, but because the betas are themselves estimated it
needs Shanken's (1992) correction.

**Testing on portfolios.** Individual stocks have betas measured with error and
very noisy returns. Sorting stocks into portfolios on characteristics spreads
the betas and averages away idiosyncratic noise. The cost, noted by Lo and
MacKinlay (1990), is data-snooping in the choice of sort.

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/plot.py:34-43` | `backtest_stats`: pyfolio `perf_stats`, which reports alpha and beta with **rf = 0**. |
| `finrl/plot.py:46-69` | `backtest_plot`: the benchmark is `^DJI` by default, a **price-weighted price index without dividends**. |
| `finrl/meta/preprocessor/yahoodownloader.py:69-72` | The strategy uses adjusted closes, so its returns are **total returns**. |
| `finrl/meta/data_processors/processor_wrds.py` | WRDS access: the route to CRSP and Ken French-style factor construction. |

## Reading the code critically

FinRL compares a total-return strategy with a price-return benchmark and leaves
out the risk-free rate. The regression intercept then picks up
$\beta \times \text{dividend yield} + (1-\beta) r_f$ of alpha that is not
skill. The Dow's dividend yield has been about 2%, and the 2023-2024 T-bill rate
about 5%, so a cash-heavy agent can show several percent of "alpha" for free.
Part C shows this in a simulation where the true alpha is zero.

## Exercises

1. *(Core)* Rerun Part C with `rf = 0.05` and `beta = 0.3`, which roughly
   describes an RL agent that holds a lot of cash. Then write a corrected
   `backtest_stats` that takes a total-return benchmark (for example `DIA`)
   and a risk-free series.
2. *(Core)* Add Shanken's (1992) correction to the Fama-MacBeth standard
   errors in Part B. How much do the t-statistics change?
3. *(Advanced)* Estimate the SDF $m_t = 1 - b^\top(f_t - E f)$ by GMM on the
   25 portfolios and test the over-identifying restrictions (Hansen's J).
   Compare with GRS.
4. *(Advanced)* Download the Fama-French 3 factors and 25 size-B/M portfolios
   from Kenneth French's data library and replicate the GRS statistics in
   Fama and French (1993, Table 9) on a modern sample.
5. *(Research)* Regress the FinRL ensemble agent's daily returns (from
   `examples/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb`) on the
   Fama-French 5 factors plus momentum, with Newey-West standard errors
   (`common.finlab.ols(..., nw_lags=5)`). Is any alpha left?

## Reading

- Cochrane (2005), *Asset Pricing*, rev. ed., ch. 1-2, 5-6, 12-16.
- Campbell (2018), *Financial Decisions and Markets*, ch. 4-5.
- Gibbons, Ross and Shanken (1989), "A Test of the Efficiency of a Given Portfolio", *Econometrica*.
- Fama and MacBeth (1973), "Risk, Return, and Equilibrium: Empirical Tests", *JPE*.
- Shanken (1992), "On the Estimation of Beta-Pricing Models", *RFS*.
- Fama and French (1993), "Common Risk Factors in the Returns on Stocks and Bonds", *JFE*.
- Hansen and Jagannathan (1991), "Implications of Security Market Data for Models of Dynamic Economies", *JPE*.
- Harvey, Liu and Zhu (2016), "... and the Cross-Section of Expected Returns", *RFS*.
