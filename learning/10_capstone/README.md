# Module 10: Capstone research projects

The capstone is a short research paper (15-25 pages) that uses this repository
as infrastructure and the methods of Modules 01-09 as the standard of evidence.
A good capstone produces a result that would survive a referee who has also
taken this course.

## What every project must include

1. **A question stated as a hypothesis**, with the economic mechanism that
   would make it true.
2. **Point-in-time data** with the survivorship and look-ahead checks from
   Module 01, documented in a data appendix.
3. **Fair benchmarks**: 1/N, a shrinkage GMV portfolio, and a simple
   model-based rule that uses the same signals as any learning method
   (Modules 03, 07 and 08).
4. **Net-of-cost results** with turnover and break-even cost (Module 06).
5. **Inference**: standard errors for every Sharpe ratio and alpha, alpha
   against a total-return benchmark and the risk-free rate (Module 04), and a
   deflated Sharpe ratio that counts every configuration you tried (Module 05).
6. **Seeds**: for any RL result, the distribution over at least 10 seeds
   (Module 08).
7. **Reproducibility**: one command that rebuilds every table and figure.

## Suggested projects

| # | Project | Builds on | Key references |
|---|---|---|---|
| 1 | **Does deep RL beat dynamic programming when DP is feasible?** Build the Module 07 predictable-returns market as a `gym.Env`, train FinRL's PPO, SAC and TD3, and measure the gap to the Bellman solution as a function of data length. | 07, 08 | Barberis (2000); Henderson et al. (2018) |
| 2 | **Reward design and risk preferences.** Train FinRL agents on the Dow 30 with the default reward, log-wealth, differential Sharpe and CRRA terminal reward. Estimate the implied risk aversion of each learned policy. | 07, 08 | Moody and Saffell (2001) |
| 3 | **Re-evaluating the ICAIF 2020 ensemble.** Rerun the ensemble notebook with point-in-time Dow membership, a total-return benchmark, costs in the portfolio environment, 10 seeds and a deflated Sharpe ratio. Is the original conclusion robust? | 01, 04, 05, 06 | Yang et al. (2020); Bailey et al. (2017) |
| 4 | **Covariance state representation.** Replace the raw 252-day covariance in `StockPortfolioEnv` with Ledoit-Wolf, a factor-model covariance or eigenvalue-clipped estimates, and fix the softmax action space. Which change matters? | 02, 03, 08 | Ledoit and Wolf (2004); DeMiguel et al. (2009) |
| 5 | **Turbulence-based de-risking as an option.** Treat the turbulence liquidation rule as a regime-timing strategy, and compare it with volatility targeting (Moreira and Muir 2017), including liquidation costs. | 02, 06 | Kritzman and Li (2010); Moreira and Muir (2017) |
| 6 | **Tax-aware RL.** Fix and test the `sr_taxlots_2` ledger, build a taxed environment, and measure the after-tax value of a learned harvesting policy against HIFO plus threshold harvesting. | 09, 08 | Constantinides (1983); Dammon, Spatt and Zhang (2001) |
| 7 | **Crypto versus equities.** Use `env_cryptocurrency_trading/` and the ccxt processor to test whether RL performance differs in a 24/7 market with higher volatility and fatter tails, with Module 05 inference. | 01, 05, 08 | Liu and Tsyvinski (2021) |
| 8 | **Factor exposures of RL agents.** Regress the returns of every agent in the examples on Fama-French 5 factors plus momentum. Are RL "alphas" disguised factor tilts? | 04, 08 | Fama and French (2015); Carhart (1997) |

## Suggested timeline (10 weeks)

| Week | Milestone |
|---|---|
| 1 | One-page proposal: question, mechanism, data, benchmarks. |
| 2-3 | Data pipeline with bias checks, plus a replication of the baseline. |
| 4-6 | Main experiments. |
| 7 | Robustness: subperiods, costs, seeds, alternative benchmarks. |
| 8 | Inference: deflated Sharpe ratio and corrected standard errors. |
| 9 | Draft, presented to peers as a referee exercise. |
| 10 | Final paper and reproducibility package. |

## How to write it up

Follow the structure of a *Journal of Finance* or *Review of Financial Studies*
empirical paper: introduction (question, answer, why it matters,
contribution), data, method, results, robustness, conclusion. Cochrane's
"Writing Tips for Ph.D. Students" is the standard short guide. Put every
number in the abstract in a table.
