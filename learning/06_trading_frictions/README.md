# Module 06: Trading frictions and execution

A signal is only worth what survives trading costs. This module covers how
costs scale with turnover, how they change the optimal policy (partial
adjustment, no-trade regions), and the classic optimal execution problem. It
then checks FinRL's environments against these ideas.

**Prerequisites:** Modules 03 and 05, dynamic programming basics.
**Lab:** `python learning/06_trading_frictions/lab.py`

## Learning objectives

1. Decompose trading costs into commissions, spread, temporary impact and
   permanent impact, and recognise the linear and square-root impact laws.
2. Compute turnover, cost drag and break-even costs for a strategy.
3. Explain why proportional costs create a no-trade region and quadratic costs
   create partial adjustment towards an "aim" portfolio.
4. Derive the Almgren-Chriss optimal liquidation schedule and its efficient
   frontier of expected cost against risk.

## Theory

**Cost of a rebalance.** Rebalancing from drifted weights $\tilde w_{t}$ to
target $w_t$ costs about $c\,\|w_t - \tilde w_t\|_1$ with proportional cost
$c$. Annual drag is $c \times$ (one-way turnover per year). The break-even cost
is the gross alpha divided by turnover. Novy-Marx and Velikov (2016) show that
many published anomalies fall below realistic break-even costs.

**Impact.** Empirically, the price impact of a metaorder of size $Q$ grows like
$\sigma\sqrt{Q/V}$, the square-root law (Tóth et al. 2011; Frazzini, Israel and
Moskowitz 2018). Almgren and Chriss (2000) use a linear model instead, with
temporary impact $\eta v$ and permanent impact $\gamma Q$, which gives closed
forms.

**Optimal liquidation.** To sell $X$ shares over $N$ periods, minimise
$E[\text{cost}] + \lambda\,\text{Var}[\text{cost}]$. The optimal holdings are

$$x_k = X\,\frac{\sinh(\kappa (T - t_k))}{\sinh(\kappa T)},\qquad
\kappa^2 \approx \frac{\lambda\sigma^2}{\eta}.$$

As $\lambda\to 0$ this becomes TWAP. A higher $\lambda$ front-loads the
selling.

**Dynamic trading with costs.** With quadratic costs and mean-reverting
signals, Gârleanu and Pedersen (2013) show that the optimal policy trades a
constant fraction of the way towards an aim portfolio. The aim is a weighted
average of current and expected future Markowitz portfolios, so it gives more
weight to slow signals. With proportional costs the optimum is a no-trade band
(Davis and Norman 1990; Constantinides 1986).

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:89` | `transaction_cost_pct` is stored and **never used**. |
| `finrl/meta/env_portfolio_allocation/env_portfolio.py:183-188` | Daily return $\sum_i w_i R_i$: implicit daily rebalancing to target at no cost. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:102-213` | `_sell_stock` and `_buy_stock`: proportional costs `buy_cost_pct` and `sell_cost_pct`, fills at the close, no impact. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:303-306` | Actions scaled by `hmax` and truncated to whole shares, so the dollar trade cap differs by stock price. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:307-309` | Turbulence rule: sell everything at once, the most expensive way to de-risk. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:315-329` | Sells are executed before buys so the cash constraint binds correctly. |
| `finrl/meta/env_stock_trading/env_stocktrading_cashpenalty.py:246-256` | A reward that penalises running out of cash. |

## Reading the code critically

- **Free trading.** In the portfolio environment the RL agent's daily
  re-weighting is free. Part A shows a policy that resamples its tilts every
  day turning over about 30% a day. At 10 bp that costs about 8% a year, which
  the environment never charges.
- **No impact.** Every order fills at the close in any size. For the Dow 30
  with `hmax = 100` that is a reasonable approximation. At `initial_amount`
  in the hundreds of millions it is not.
- **Fire sales.** The turbulence rule (Module 02) liquidates everything in one
  step, which is the opposite of the Almgren-Chriss solution.

## Exercises

1. *(Core)* Add costs to `StockPortfolioEnv.step`. Track the drifted weights
   and charge `transaction_cost_pct` times turnover. Keep it as a local
   experiment; do not change the library on master.
2. *(Core)* In Part B, find the cost at which the best speed is 0.2, and relate
   it to the Gârleanu-Pedersen trading rate formula.
3. *(Advanced)* Replace the linear temporary impact in Part C with a
   square-root law and solve for the optimal schedule numerically.
4. *(Advanced)* Implement a no-trade band policy for proportional costs in
   Part B and compare it with partial adjustment.
5. *(Research)* Add square-root impact to `StockTradingEnv`, retrain PPO, and
   measure how the learned policy's turnover changes with
   `initial_amount` from \$1m to \$1bn.

## Reading

- Almgren and Chriss (2000), "Optimal Execution of Portfolio Transactions", *Journal of Risk*.
- Gârleanu and Pedersen (2013), "Dynamic Trading with Predictable Returns and Transaction Costs", *JF*.
- Constantinides (1986), "Capital Market Equilibrium with Transaction Costs", *JPE*.
- Frazzini, Israel and Moskowitz (2018), "Trading Costs", working paper (AQR).
- Novy-Marx and Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs", *RFS*.
- Bouchaud, Bonart, Donier and Gould (2018), *Trades, Quotes and Prices*, ch. 11-12.
- Hasbrouck (2007), *Empirical Market Microstructure*.
