# Learning finance with FinRL

A graduate course in empirical finance, portfolio choice and reinforcement
learning for trading. It uses the FinRL codebase as a set of worked examples:
each module teaches the finance and econometrics at masters or first-year PhD
level, then reads the relevant FinRL code critically, line by line.

Nothing in `finrl/` is modified. Everything for the course lives in this folder.

## How a module works

Every module folder contains:

- **`README.md`**: learning objectives, the theory with the key results
  derived or stated, a table of the FinRL files and line numbers that put the
  theory into practice, a critical reading of that code, graded exercises
  (*core*, *advanced*, *research*) and a reading list.
- **`lab.py`**: a runnable lab that simulates a market with known parameters,
  reproduces the relevant FinRL logic, and checks the numbers against theory
  with assertions. Run it, read the output, then change it.

The labs need only numpy, pandas and scipy, not FinRL's own dependencies, which
are pinned to early 2023 and are hard to install today. Simulated data with
known parameters lets each lab compare an estimate with the truth, which real
data never allows. The exercises then move to real data and FinRL's notebooks.

```bash
pip install -r learning/requirements.txt
python learning/run_all_labs.py          # all labs, about a minute
python learning/03_portfolio_theory/lab.py  # one lab
```

## Syllabus

| # | Module | Finance and econometrics | FinRL code it reads |
|---|---|---|---|
| 01 | [Financial data and its biases](01_financial_data/) | Returns, adjustment, survivorship, delisting, look-ahead | Yahoo downloader, `FeatureEngineer.clean_data`, the fill step |
| 02 | [Returns, risk and covariance](02_risk_and_covariance/) | Mahalanobis distance, Hotelling $T^2$, Ledoit-Wolf shrinkage | Turbulence index, ensemble threshold, `cov_list` state |
| 03 | [Portfolio theory](03_portfolio_theory/) | Frontier, tangency, two-fund separation, 1/N versus estimated MV | `StockPortfolioEnv` softmax actions, Markowitz baseline |
| 04 | [Asset pricing and factor models](04_asset_pricing/) | SDF, HJ bound, GRS test, Fama-MacBeth, Shanken | `backtest_stats` and `backtest_plot` alpha and beta |
| 05 | [Performance evaluation](05_performance_evaluation/) | Sharpe standard errors, multiple testing, deflated Sharpe | Ensemble model selection, `get_validation_sharpe` |
| 06 | [Trading frictions](06_trading_frictions/) | Turnover, break-even costs, partial adjustment, Almgren-Chriss | Cost handling in both environments, `hmax`, turbulence liquidation |
| 07 | [Dynamic portfolio choice](07_dynamic_portfolio_choice/) | Merton, HJB, Bellman, hedging demand under predictability | `StockTradingEnv` as an MDP: state, action, transition, reward |
| 08 | [RL for trading](08_rl_for_trading/) | Q-learning versus DP, sample efficiency, seed dispersion | `DRLAgent`, `DRLEnsembleAgent`, SB3/ElegantRL/RLlib wrappers |
| 09 | [Tax-aware investing](09_tax_aware_investing/) | Tax lots, deferral, tax-loss harvesting | The unmerged `sr_taxlots_2` ledger, and a bug in it |
| 10 | [Capstone](10_capstone/) | Research design and writing | Your choice of eight projects |

Modules 01-05 form an empirical asset pricing sequence, in the tradition of
Chicago Booth's PhD core (Cochrane, Fama-French). Modules 06-09 cover
portfolio choice and trading, where the RL material sits. A one-semester
course covers 01-08. Module 09 and the capstone extend it.

## What the course finds in the code

Reading the code critically is part of the method, and the labs reproduce each
of these findings. They are teaching points, not patches. The library is
unchanged, and fixing them is left to the exercises.

| Module | Finding | Where |
|---|---|---|
| 01 | `ffill().bfill()` on the long panel fills across tickers and from the future | `finrl/meta/preprocessor/preprocessors.py:106` |
| 01 | `close` is adjusted but `high` and `low` are not, and `cci_30` and `dx_30` mix them | `finrl/meta/preprocessor/yahoodownloader.py:69-72` |
| 01 | `clean_data` and the ticker lists keep only survivors | `preprocessors.py:109-134`, `config_tickers.py:7` |
| 02 | The ensemble's turbulence threshold if/else is dead code; only the 0.99 quantile is used | `finrl/agents/stablebaselines3/models.py:401-416` |
| 03 | Softmax over actions in [0, 1] limits every weight to 1.3%-8.6% for 30 stocks | `finrl/meta/env_portfolio_allocation/env_portfolio.py:96, 225-229` |
| 04 | Alpha against a price index with rf = 0 adds $\beta\cdot$yield $+ (1-\beta) r_f$ | `finrl/plot.py:34-69` |
| 05 | Model choice on a 63-day validation Sharpe (standard error about 2.0), annualised with $\sqrt 4$ | `models.py:214-230` |
| 06 | `transaction_cost_pct` is stored and never charged | `env_portfolio.py:89` |
| 07 | The reward is the change in wealth, so the agent is risk-neutral | `finrl/meta/env_stock_trading/env_stocktrading.py:349-351` |
| 09 | `tax_used` is overwritten instead of accumulated | `origin/sr_taxlots_2:finrl/env/accounting/ledger.py:224` |

## Core texts

These are standard in top PhD programmes and are cited across the modules.

- Cochrane, *Asset Pricing* (rev. ed., 2005). SDF approach (Modules 03-04).
- Campbell, Lo and MacKinlay, *The Econometrics of Financial Markets* (1997). Empirical methods (01, 04, 05).
- Campbell, *Financial Decisions and Markets* (2018). Portfolio choice and asset pricing (03, 04, 07).
- Campbell and Viceira, *Strategic Asset Allocation* (2002). Long-horizon portfolio choice (07).
- Duffie, *Dynamic Asset Pricing Theory* (3rd ed., 2001). Continuous-time theory (07).
- López de Prado, *Advances in Financial Machine Learning* (2018). Backtesting (01, 05).
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018). RL (08).

## Prerequisites

Probability and statistics at the level of Casella and Berger, linear algebra,
multivariable calculus, and working Python with numpy and pandas. Stochastic
calculus helps for Module 07 but is not required for the lab.

## Using the FinRL notebooks alongside

Several exercises run the original pipelines in `examples/`. Those need FinRL's
full dependency set (`pip install -e .` from the repo root). Expect to update
some pins, since the snapshot is from February 2023. Do the work in a separate
virtual environment so the labs keep working regardless.
