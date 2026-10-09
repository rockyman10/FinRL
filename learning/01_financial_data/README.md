# Module 01: Financial data and its biases

Before you estimate anything, you have to know what the numbers mean, when you
could have known them, and which observations are missing. Most published
backtests that fail out of sample fail here, not in the model.

**Prerequisites:** probability, basic pandas. **Lab:** `python learning/01_financial_data/lab.py`

## Learning objectives

1. Define simple and log returns, total return versus price return, and explain
   when each aggregates correctly (across assets versus across time).
2. Explain price adjustment for splits and dividends, and why adjusted and
   unadjusted fields must never be mixed.
3. Identify and quantify survivorship bias, delisting bias and look-ahead bias.
4. Read a data pipeline critically and find where those biases enter.

## Theory

**Returns.** The simple return $R_t = P_t/P_{t-1} - 1$ aggregates across assets
($R_{p,t} = \sum_i w_i R_{i,t}$). The log return $r_t = \ln(1+R_t)$ aggregates
across time ($r_{t\to t+k} = \sum_{j} r_{t+j}$). Mixing them is a common source
of small, persistent errors. Total return includes dividends:
$R_t = (P_t + D_t)/P_{t-1} - 1$. An "adjusted close" series is constructed so
that its simple returns equal total returns. Its level is fiction, but its
returns are correct.

**Point-in-time data.** A dataset is point-in-time if every value is stamped
with the date it became *known*, not the date it refers to. Prices are nearly
point-in-time. Accounting data, index membership and revisions are not, so a
naive merge leaks the future. In CRSP/Compustat work this is the reason for the
standard six-month lag between fiscal year-end and portfolio formation
(Fama and French 1992).

**Survivorship and delisting.** If the sample keeps only firms that exist at
the end, it drops the losers, and average returns are biased upward
(Brown, Goetzmann, Ibbotson and Ross 1992). Even when delisted firms are kept,
their final return is often missing. Shumway (1997) shows that delisting
returns for performance-related delistings average about -30%, and ignoring
them biases small-cap returns upward.

**Look-ahead.** A feature at time $t$ must be measurable with respect to the
information set $\mathcal{F}_t$. Common violations include centred moving
averages, normalising with full-sample statistics, back-filling, and using
today's close to trade at today's open.

## Where it lives in FinRL

| File | What to read |
|---|---|
| `finrl/meta/preprocessor/yahoodownloader.py:36-87` | Downloads OHLCV, then **replaces `close` with adjusted close** (lines 69-72) while leaving open/high/low raw. |
| `finrl/meta/preprocessor/preprocessors.py:109-134` | `clean_data`: pivots closes and `dropna(axis=1)`, which keeps only full-history tickers. |
| `finrl/meta/preprocessor/preprocessors.py:136-170` | Technical indicators via `stockstats`, computed per ticker. |
| `finrl/meta/preprocessor/preprocessors.py:106` | `ffill().bfill()` on the long panel. |
| `finrl/meta/data_processors/` | Alternative sources (Alpaca, WRDS, ccxt). `processor_wrds.py` is the closest to research-grade data. |
| `finrl/config_tickers.py:7` | `DOW_30_TICKER`, the *current* constituents. |
| `finrl/config.py:10-17` | Train/test/trade date splits. |

## Reading the code critically

The lab demonstrates each of these:

- **Survivorship.** `clean_data` keeps only tickers with no missing closes,
  and the ticker lists are today's index members. Both remove failures.
- **Mixed adjustment.** `close` is adjusted while `high` and `low` are not.
  `cci_30` and `dx_30`, two of the default `INDICATORS`, combine all three.
- **The fill step.** `fillna(method="ffill").fillna(method="bfill")` runs on a
  frame sorted by `(date, tic)`. That fills one ticker's gap with a neighbour's
  value and back-fills warm-up NaNs from the future.
- **Fine as written.** The environment's timing (observe close $t$, trade at
  close $t$, earn close $t+1$) is correct. Rolling indicators are causal.

## Exercises

1. *(Core)* Rewrite the fill step so that it is causal and per-ticker. Show on
   the toy frame from the lab that it no longer leaks.
2. *(Core)* Rerun Part A with `delisting_return=0` (the "missing delisting
   return" case) and with `delist_below=0.5`. How does the bias scale?
3. *(Advanced)* Write a point-in-time Dow 30 universe: a table of
   `(ticker, start, end)` membership spells. Make `clean_data` accept it and
   keep a stock only while it is a member. Sources: S&P/Dow Jones press
   releases, or CRSP `dsp500list` via `processor_wrds.py` if you have WRDS.
4. *(Advanced)* Total versus price return: download `^DJI` and `DIA` for the
   same period. Annualise the gap and relate it to the dividend yield. This
   matters again in Module 04.
5. *(Research)* Replicate Shumway (1997, Table III) style delisting-return
   imputation on CRSP and measure its effect on a small-cap anomaly.

## Reading

- Campbell, Lo and MacKinlay (1997), *The Econometrics of Financial Markets*, ch. 1.
- Brown, Goetzmann, Ibbotson and Ross (1992), "Survivorship Bias in Performance Studies", *RFS*.
- Shumway (1997), "The Delisting Bias in CRSP Data", *JF*.
- Fama and French (1992), "The Cross-Section of Expected Stock Returns", *JF*, section I (data timing).
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 2-4.
