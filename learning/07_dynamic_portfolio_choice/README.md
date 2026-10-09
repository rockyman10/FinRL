# Module 07: Dynamic portfolio choice and stochastic control

Reinforcement learning for trading is a numerical method for a problem finance
solved analytically in special cases half a century ago: choosing a portfolio
over time to maximise expected utility. This module builds that theory and then
reads FinRL's environment as the Markov decision process it really is.

**Prerequisites:** Modules 03 and 06, Itô calculus basics, dynamic programming.
**Lab:** `python learning/07_dynamic_portfolio_choice/lab.py`

## Learning objectives

1. Solve Merton's problem with the HJB equation, and explain when investors
   are myopic.
2. State the Bellman equation for discrete-time portfolio choice with a state
   variable, and solve it numerically by backward induction with quadrature.
3. Explain intertemporal hedging demand and the effect of return
   predictability on long-horizon allocation.
4. Write FinRL's stock trading environment as an MDP $(S, A, P, r, \gamma)$,
   and say what objective its reward implies.

## Theory

**Merton (1969, 1971).** With wealth dynamics
$dW = W[(r + \pi\mu)dt + \pi\sigma dB]$ and CRRA utility
$u(W) = W^{1-\gamma}/(1-\gamma)$, the HJB equation gives

$$\pi^* = \frac{\mu}{\gamma\sigma^2}.$$

The horizon does not appear: with i.i.d. returns and CRRA utility, the
long-horizon investor behaves myopically. Log utility ($\gamma = 1$) gives the
Kelly criterion, which maximises the growth rate $E[\log W]$.

**State variables and hedging.** If expected returns depend on a state $x_t$,
the value function is $V_t(W, x) = \frac{W^{1-\gamma}}{1-\gamma}\psi_t(x)$ and

$$\psi_t(x) = \operatorname*{opt}_{\pi}\ E_t\!\left[\big(1 + r_f + \pi r^e_{t+1}\big)^{1-\gamma}\,\psi_{t+1}(x_{t+1})\right].$$

The optimal $\pi$ is the myopic demand plus a hedging demand proportional to
the covariance between returns and innovations in $x$ (Merton 1973). With
mean reversion ($\rho < 0$) and $\gamma > 1$, long-horizon investors hold more
stock (Kim and Omberg 1996; Campbell and Viceira 1999; Barberis 2000). Barberis
also shows that parameter uncertainty can reverse this.

**The MDP view.** An MDP is a state space $S$, actions $A$, transitions
$P(s'|s,a)$, rewards $r(s,a,s')$ and a discount factor. The Bellman equation
above is the finance special case. RL methods (Module 08) solve it by sampling
transitions instead of integrating against a known $P$. That is useful when
$P$ is unknown or the state is high-dimensional, and costly in samples
otherwise.

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/meta/env_stock_trading/env_stocktrading.py:397-451` | **State**: `[cash, prices..., shares..., indicators...]`. Wealth is in it, the predictor is in the indicators. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:60` | **Action**: `Box(-1, 1)` per stock, scaled to share counts by `hmax`. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:331-341` | **Transition**: move to the next day's data. Prices are exogenous, so the agent never moves the market. |
| `finrl/meta/env_stock_trading/env_stocktrading.py:349-351` | **Reward**: change in total asset value, times `reward_scaling`. |
| `finrl/agents/stablebaselines3/models.py:70-100` | The discount factor and other hyperparameters passed to SB3 (`gamma`, default 0.99). |
| `finrl/meta/env_stock_trading/env_stocktrading_cashpenalty.py:246-256` and `env_stocktrading_stoploss.py:255-290` | Alternative rewards: return per day elapsed with penalties for low cash, stop-loss breaches and low profit. |

## Reading the code critically

- **The reward is risk-neutral.** Summed over an episode, the reward is
  $W_T - W_0$ (times a constant). The agent maximises expected dollar wealth,
  which under i.i.d. returns means maximum leverage. `hmax` and the cash
  constraint are the only reasons it does not. Part B shows the outcome: the
  highest mean wealth, and a 15% chance of losing half.
- **Discounting.** SB3's discount $\gamma = 0.99$ per *day* puts a half-life of
  about 69 trading days on the reward. That is a modelling choice with no
  economic justification, and it interacts with the risk-neutral reward.
- **Better rewards.** A per-step reward of $\log(W_{t+1}/W_t)$ gives log utility
  exactly, since the sum telescopes. CRRA needs a terminal reward or a
  differential approximation. Moody and Saffell (2001) propose the
  "differential Sharpe ratio" as a per-step reward.

## Exercises

1. *(Core)* Derive Merton's $\pi^*$ from the HJB equation. Then show that
   Part A's closed-form certainty equivalent is correct.
2. *(Core)* Set $\rho = 0$ in Part C. What happens to the hedging demand, and
   why?
3. *(Advanced)* Add a proportional transaction cost to Part C, with the
   current risky share as a second state variable. Plot the no-trade region
   (cf. Balduzzi and Lynch 1999).
4. *(Advanced)* Modify `StockTradingEnv` in a local copy so that its reward is
   $\log(W_{t+1}/W_t)$. Retrain PPO on the Dow 30 and compare leverage, cash
   holdings and drawdowns with the original reward.
5. *(Research)* Estimate the Part C model on the S&P 500 with the dividend
   yield as predictor, with and without parameter uncertainty (Barberis 2000).
   Does the hedging demand survive?

## Reading

- Merton (1969), "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case", *REStat*.
- Merton (1973), "An Intertemporal Capital Asset Pricing Model", *Econometrica*.
- Campbell and Viceira (2002), *Strategic Asset Allocation*, ch. 2-4.
- Barberis (2000), "Investing for the Long Run When Returns Are Predictable", *JF*.
- Kim and Omberg (1996), "Dynamic Nonmyopic Portfolio Behavior", *RFS*.
- Brandt (2010), "Portfolio Choice Problems", *Handbook of Financial Econometrics*.
- Pham (2009), *Continuous-time Stochastic Control and Optimization with Financial Applications*.
- Moody and Saffell (2001), "Learning to Trade via Direct Reinforcement", *IEEE TNN*.
