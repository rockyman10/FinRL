# Module 05: Performance evaluation and backtest statistics

A backtest is an estimate, and every reported number has a standard error. This
module covers the sampling theory of the Sharpe ratio, the inflation that comes
from trying many strategies, and how FinRL's ensemble agent picks a model.

**Prerequisites:** Module 04, asymptotic theory (delta method), order statistics.
**Lab:** `python learning/05_performance_evaluation/lab.py`

## Learning objectives

1. Derive the asymptotic standard error of the Sharpe ratio under i.i.d.
   normal returns and under skewness and kurtosis.
2. Explain why the maximum of many Sharpe ratios is biased upward. Compute the
   probabilistic and deflated Sharpe ratios.
3. Define Sortino, Calmar and maximum drawdown, and know what each ignores.
4. Judge whether a model-selection rule has enough statistical power for its
   decisions to mean anything.

## Theory

**Sampling error of the Sharpe ratio.** With i.i.d. normal returns and per-period
$SR$, the delta method gives $\text{Var}(\widehat{SR}) \approx (1 + SR^2/2)/T$
(Lo 2002). With skewness $\gamma_3$ and kurtosis $\gamma_4$ (Mertens 2002):

$$\text{Var}(\widehat{SR}) \approx \frac{1}{T}\left(1 + \frac{SR^2}{2} - \gamma_3 SR + \frac{\gamma_4 - 3}{4}SR^2\right).$$

Annualised, the standard error is roughly $1/\sqrt{\text{years}}$. A strategy
with a true Sharpe of 0.5 needs about 16 years of data to reach $t=2$.
Autocorrelated returns need Lo's (2002) correction to the $\sqrt{q}$ scaling as
well.

**Multiple testing.** If $N$ independent strategies have zero true Sharpe,
the largest estimate is approximately (Bailey and López de Prado 2014)

$$E[\max_n \widehat{SR}_n] \approx \sqrt{V}\left((1-\gamma)\Phi^{-1}\!\left(1-\tfrac1N\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right),$$

where $\gamma$ is the Euler-Mascheroni constant and $V$ the variance of the
estimates. The deflated Sharpe ratio is the probabilistic Sharpe ratio
$\Phi\big((\widehat{SR} - SR_0)/\hat\sigma_{SR}\big)$ evaluated at that
$SR_0$. Harvey, Liu and Zhu (2016) apply the same reasoning to factor discovery
and argue for $t > 3$.

**Drawdown statistics** depend on the path and on the sample length. A longer
backtest has a deeper expected maximum drawdown even with identical skill, so
Calmar ratios cannot be compared across horizons.

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/plot.py:34-43` | `backtest_stats` delegates to pyfolio `perf_stats`. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:246-251` | End-of-episode Sharpe with $\sqrt{252}$. |
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:145-153` | The same for the portfolio environment, with no risk-free rate. |
| `finrl/agents/stablebaselines3/models.py:214-230` | `get_validation_sharpe`: annualises daily returns with $\sqrt{4}$, and returns `inf` when variance is zero and the mean is positive. |
| `finrl/agents/stablebaselines3/models.py:327-700` | `run_ensemble_strategy`: each quarter, picks among A2C, PPO and DDPG on a 63-day validation Sharpe (`rebalance_window = validation_window = 63` in the ICAIF 2020 notebook). |
| `finrl/agents/stablebaselines3/tune_sb3.py:131` | Hyperparameter tuning (Optuna) that maximises the Sharpe ratio: another search that needs deflating. |

## Reading the code critically

- **Selection on noise.** The standard error of an annualised Sharpe ratio
  over 63 days is about 2.0. Choosing between models whose true Sharpe ratios
  differ by 0.2 on that basis is close to random. Part C shows the ensemble
  picks the best of three models only slightly more often than chance.
- **Hidden trials.** Each notebook run that tries a hyperparameter, ticker
  universe or date split adds to the effective number of trials $N$, and
  none of them are recorded. Bailey et al. (2014) call this "backtest
  overfitting".
- **Units.** $\sqrt{4}$ applied to daily data gives a number that is neither
  daily nor annual. It does not change the ranking, but it ends up in logs and
  plots as if it were a Sharpe ratio.

## Exercises

1. *(Core)* Derive Lo's i.i.d. standard error with the delta method, starting
   from the joint asymptotic distribution of $(\hat\mu, \hat\sigma^2)$.
2. *(Core)* In Part C, vary the validation window from 21 to 504 days and plot
   how often the best model is picked. How long a window would it take to pick
   it 80% of the time?
3. *(Advanced)* Implement Combinatorially Symmetric Cross-Validation and the
   Probability of Backtest Overfitting (Bailey, Borwein, López de Prado and
   Zhu 2017) for the 200 strategies in Part B.
4. *(Advanced)* Use the Ledoit and Wolf (2008) robust test to check whether
   two strategies' Sharpe ratios differ, allowing for correlated, heavy-tailed
   returns.
5. *(Research)* Count every configuration tried in `tune_sb3.py` for one
   study, then report the deflated Sharpe ratio of the chosen agent's backtest.

## Reading

- Lo (2002), "The Statistics of Sharpe Ratios", *FAJ*.
- Mertens (2002), "Comments on Variance of the IID Estimator in Lo (2002)", working paper.
- Bailey and López de Prado (2014), "The Deflated Sharpe Ratio", *Journal of Portfolio Management*.
- Bailey, Borwein, López de Prado and Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism", *Notices of the AMS*.
- Harvey, Liu and Zhu (2016), "... and the Cross-Section of Expected Returns", *RFS*.
- Ledoit and Wolf (2008), "Robust Performance Hypothesis Testing with the Sharpe Ratio", *JEF*.
- White (2000), "A Reality Check for Data Snooping", *Econometrica*.
