# Module 09: Tax-aware investing

For a taxable investor, after-tax return is the only return that matters, and
the tax code turns a static problem into a path-dependent one. Each purchase
creates a tax lot with its own basis and holding period, and the order in which
lots are sold changes the bill. The repository has an unfinished attempt at
this on two side branches, which makes a good exercise in reading and testing
someone else's financial code.

**Prerequisites:** Modules 06-07.
**Lab:** `python learning/09_tax_aware_investing/lab.py`

## Learning objectives

1. Explain tax lots, cost basis, short-term versus long-term holding periods,
   and lot-relief methods (FIFO, LIFO, HIFO, specific identification).
2. Model the value of tax deferral and of tax-loss harvesting, and explain
   where that value comes from.
3. Explain the wash-sale rule and why harvesting needs substitute securities.
4. Write tests that find bugs in a lot-accounting ledger.

## Theory

**Deferral.** A gain realised later is taxed later. At return $r$ and tax rate
$\tau$ over $T$ years, deferring until the end gives
$(1+r)^T(1-\tau) + \tau$ instead of $(1 + r(1-\tau))^T$. The gap grows with $T$
and $r$. At death, US basis steps up and the deferred tax disappears.

**The realisation option.** Constantinides (1983) shows that with
asymmetric rates the investor holds an option on when to realise. Losses
should be realised at once, short-term if possible, and gains deferred until
they qualify as long-term. Under some rate structures it can even pay to
realise long-term gains to reset the holding period (Constantinides 1984).

**Tax-loss harvesting.** Selling a lot below basis and buying a close
substitute books a loss that offsets other gains, at up to 37% in the US. The
lower basis means a larger gain later, often taxed at 20%. Both deferral and
rate arbitrage add value. Berkin and Ye (2003) estimate gains of tens of basis
points a year, larger early on and with more dispersion across holdings, which
is why direct indexing harvests stock by stock. Wash-sale rules disallow the
loss if a "substantially identical" security is bought within 30 days.

**Dynamic programming with taxes.** The state must include every lot's basis
and age. That curse of dimensionality is why the literature relies on
approximations such as average basis (Dammon, Spatt and Zhang 2001) and why RL
is a plausible tool here.

## Where it lives in FinRL

The work is on two upstream branches from 2021 that were never merged:

| Location | What to read |
|---|---|
| `origin/sr_tax_lots`, `origin/sr_taxlots_2` | Run `git log --oneline origin/sr_taxlots_2` and look for "Working ledger implementation ... Computation of tax lots". |
| `origin/sr_taxlots_2:finrl/env/accounting/ledger.py:50-105` | `log_transactions`: records buys and sells per date, then calls `compute_tax_lots` for each sale. |
| `origin/sr_taxlots_2:finrl/env/accounting/ledger.py:107-141` | `_compute_holdings`: splits holdings into long-term and short-term. |
| `origin/sr_taxlots_2:finrl/env/accounting/ledger.py:188-240` | `compute_tax_lots`: FIFO consumption of lots, with the 365-day threshold. |

Read them with `git show origin/sr_taxlots_2:finrl/env/accounting/ledger.py`.
The branches predate the current layout (`finrl/env/` rather than
`finrl/meta/`), so they do not merge cleanly into master.

## Reading the code critically

- **Bug: lots forget earlier sales.** Line 224 sets
  `a_data[date]["tax_used"] = shares_consumed` where it should add to it. After
  a lot has been partly used twice, the ledger lets it be sold again. Part C
  reproduces this with three sales.
- **Shallow copy.** `copy(self.d[asset])` copies the outer dict only, so the
  "copy" shares the per-date dicts it modifies. That hides the intent (a
  transactional update) and would break a rollback.
- **Holding period.** `sell_date - 365 days > buy_date` is close to the US
  rule ("more than one year"), but leap years and the exact day matter at the
  boundary.
- **Never wired into a reward.** The ledger computes gains, but no
  environment taxes them, so no agent ever learns about taxes.

## Exercises

1. *(Core)* Fix the `tax_used` bug in the lab's copy, and turn Part C into a
   `pytest` test that fails before the fix and passes after.
2. *(Core)* Add specific identification that minimises tax: choose lots to
   minimise current tax given the short-term and long-term rates. When does
   it differ from HIFO?
3. *(Advanced)* Add a 30-day wash-sale rule to Part B: after a harvest, the
   investor holds a substitute with tracking error $\sigma_{TE}$. How does the
   value of harvesting change with $\sigma_{TE}$?
4. *(Advanced)* Write a tax-aware version of `StockTradingEnv` in a local
   copy: plug in a corrected ledger and charge tax on realised gains at each
   step. Train PPO with and without taxes, and compare turnover.
5. *(Research)* Use Dammon, Spatt and Zhang (2001) with average basis as a
   benchmark for an RL agent trained in exercise 4.

## Reading

- Constantinides (1983), "Capital Market Equilibrium with Personal Tax", *Econometrica*.
- Constantinides (1984), "Optimal Stock Trading with Personal Taxes", *JFE*.
- Dammon, Spatt and Zhang (2001), "Optimal Consumption and Investment with Capital Gains Taxes", *RFS*.
- Berkin and Ye (2003), "Tax Management, Loss Harvesting, and HIFO Accounting", *FAJ*.
- Chaudhuri, Burnham and Lo (2020), "An Empirical Evaluation of Tax-Loss Harvesting Alpha", *FAJ*.
- IRS Publication 550, *Investment Income and Expenses* (wash sales and holding periods).
