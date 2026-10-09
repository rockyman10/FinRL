# Module 08: Reinforcement learning for trading

This is the module the repository was built for. It treats deep RL as one
estimator for the dynamic programme of Module 07, with the same questions about
bias, variance and overfitting as any other estimator. The lab uses a market
where the optimum is known. The FinRL notebooks then apply the same methods to
the Dow 30.

**Prerequisites:** Modules 05-07; Sutton and Barto ch. 3-6 and 13.
**Lab:** `python learning/08_rl_for_trading/lab.py`

## Learning objectives

1. Explain value-based (Q-learning, DQN), policy-gradient (REINFORCE, A2C,
   PPO) and deterministic actor-critic (DDPG, TD3, SAC) methods, and which
   suit continuous portfolio actions.
2. Compare model-free RL with model-based dynamic programming on a problem
   with a known solution. Quantify the data needed.
3. Diagnose overfitting and seed dependence in RL backtests.
4. Read FinRL's training pipeline (environment, agent, ensemble) and critique
   its evaluation protocol.

## Theory

**From Bellman to Q-learning.** Q-learning applies stochastic approximation to
$Q(s,a) = E[r + \gamma \max_{a'} Q(s',a')]$. It needs no transition model, at
the cost of needing enough samples of every $(s,a)$. Financial data has one
history, low signal-to-noise and non-stationarity, so samples are expensive.

**Policy gradients.** $\nabla_\theta J = E[\nabla_\theta \log \pi_\theta(a|s)\,A(s,a)]$.
A2C adds a learned baseline. PPO limits each update with a clipped ratio.
DDPG, TD3 and SAC learn a deterministic or entropy-regularised policy for
continuous actions with a learned critic. All of them are high-variance
estimators, and published results are sensitive to seeds and implementation
details (Henderson et al. 2018).

**Finance-specific issues.**
- *Signal-to-noise.* Daily return predictability has an $R^2$ well under 1%.
  Learning a value function from returns resembles estimating a mean
  (Module 05), with a standard error of about $1/\sqrt{\text{years}}$.
- *Non-stationarity.* Regimes change (Module 02), so the training
  distribution is not the test distribution.
- *One history.* There is no simulator. Backtests re-use the same data,
  which inflates performance (Module 05). Synthetic markets, as in this lab,
  can help with method development but not with discovering alpha.
- *Objective.* The reward defines the risk preference (Module 07).

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/agents/stablebaselines3/models.py:48-162` | `DRLAgent`: wraps SB3 A2C, DDPG, PPO, SAC and TD3. `get_model`, `train_model`, `DRL_prediction`. |
| `finrl/agents/stablebaselines3/models.py:165-700` | `DRLEnsembleAgent`: quarterly retraining and model selection by validation Sharpe (Module 05). |
| `finrl/agents/stablebaselines3/tune_sb3.py`, `hyperparams_opt.py` | Optuna hyperparameter search. |
| `finrl/agents/elegantrl/models.py`, `finrl/agents/rllib/models.py` | The same agents through ElegantRL and Ray RLlib. |
| `finrl/config.py:35-62` | Default hyperparameters for each algorithm. |
| `finrl/train.py`, `finrl/test.py`, `finrl/trade.py` | Script entry points for train, test and trade. |
| `examples/Stock_NeurIPS2018_*.ipynb` | Data, then train, then backtest, on the Dow 30. |
| `examples/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb` | The ensemble strategy of Yang, Liu, Zhong and Walid (2020). |

## Reading the code critically

- **Evaluation protocol.** One train/test split and one seed, with the
  baseline chosen after the fact, is not enough. Part C of the lab shows that
  the out-of-sample Sharpe from one dataset ranges widely across seeds.
- **No model-based benchmark.** None of the examples compares the agent with a
  simple dynamic or mean-variance rule that uses the same signals (the
  technical indicators). Without one, you cannot tell whether RL added
  anything beyond the features.
- **Training length.** The Dow 30 example trains on 2014-01 to 2020-07
  (`config.py:10-11`), which is about 1,650 daily transitions. Its state has
  1 + 30 × (2 + 8) = 301 dimensions. Part B shows overfitting at that data
  length even with 21 states.

## Exercises

1. *(Core)* Raise `COST` to 0.003 in the lab. How does the optimal no-trade
   region change, and how many years does Q-learning now need?
2. *(Core)* Replace the risk-neutral reward with a mean-variance reward
   $\pi r - \frac{\kappa}{2}\pi^2\sigma^2$ and re-solve both the DP and
   Q-learning.
3. *(Advanced)* Write the lab's market as a `gym.Env` and train FinRL's
   `DRLAgent` (PPO) on it. Compare with the DP optimum. This is the cleanest
   test of whether the deep RL pipeline works at all.
4. *(Advanced)* Run the NeurIPS 2018 notebook pipeline with 10 seeds and
   report the distribution of test Sharpe ratios and its deflated Sharpe ratio
   (Module 05).
5. *(Research)* Add a regime-switching state (Module 02's turbulence) to the
   lab's market and test whether RL learns a regime-dependent policy that
   beats the regime-blind optimum.

## Reading

- Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed.
- Moody and Saffell (2001), "Learning to Trade via Direct Reinforcement", *IEEE TNN*.
- Kolm and Ritter (2019), "Dynamic Replication and Hedging: A Reinforcement Learning Approach", *JFDS*.
- Buehler, Gonon, Teichmann and Wood (2019), "Deep Hedging", *Quantitative Finance*.
- Henderson et al. (2018), "Deep Reinforcement Learning That Matters", *AAAI*.
- Yang, Liu, Zhong and Walid (2020), "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy", *ICAIF*.
- Liu et al. (2020), "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading", NeurIPS workshop.
- Hambly, Xu and Yang (2023), "Recent Advances in Reinforcement Learning in Finance", *Mathematical Finance*.
