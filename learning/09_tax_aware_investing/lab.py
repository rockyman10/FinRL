"""Module 09 lab: tax lots, lot relief methods, tax-loss harvesting, and a ledger bug.

Run:  python learning/09_tax_aware_investing/lab.py

Parts
  A. FIFO versus HIFO lot relief on the same trades.
  B. The value of tax-loss harvesting (deferral and rate arbitrage).
  C. Reading the unfinished ledger on the `sr_taxlots_2` branch: a lot-tracking bug.
"""
from __future__ import annotations

import datetime
import pathlib
import sys
from copy import copy

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.finlab import TRADING_DAYS, header  # noqa: E402

ST_RATE, LT_RATE = 0.37, 0.20  # US top federal rates: short-term (ordinary) and long-term gains
rng = np.random.default_rng(99)


class Lots:
    """Minimal tax-lot ledger for one asset."""

    def __init__(self, method: str):
        self.method = method
        self.lots: list[dict] = []  # each: {"date", "shares", "price"}

    def buy(self, date: int, shares: float, price: float) -> None:
        self.lots.append({"date": date, "shares": shares, "price": price})

    def sell(self, date: int, shares: float, price: float) -> tuple[float, float]:
        """Returns (short-term gain, long-term gain). Dates are trading-day indices."""
        order = {
            "FIFO": lambda lot: lot["date"],
            "LIFO": lambda lot: -lot["date"],
            "HIFO": lambda lot: -lot["price"],
        }[self.method]
        st = lt = 0.0
        for lot in sorted(self.lots, key=order):
            if shares <= 0:
                break
            q = min(shares, lot["shares"])
            gain = q * (price - lot["price"])
            if date - lot["date"] > TRADING_DAYS:
                lt += gain
            else:
                st += gain
            lot["shares"] -= q
            shares -= q
        self.lots = [lot for lot in self.lots if lot["shares"] > 1e-12]
        return st, lt


def tax_on(st: float, lt: float) -> float:
    """Tax with losses netted (simplified; losses assumed usable against other gains)."""
    return ST_RATE * st + LT_RATE * lt


# ---------------------------------------------------------------------------
header("A. Lot relief: same trades, different tax bills")
# ---------------------------------------------------------------------------
T = TRADING_DAYS * 6
price = 100 * np.exp(np.cumsum(rng.normal(0.07 / TRADING_DAYS, 0.20 / np.sqrt(TRADING_DAYS), T)))
rows = []
for method in ["FIFO", "LIFO", "HIFO"]:
    book, taxes = Lots(method), []
    for t in range(T):
        if t % 21 == 0:  # buy $1,000 a month
            book.buy(t, 1000 / price[t], price[t])
        if t % 63 == 62:  # sell 20% of the position each quarter
            shares = sum(lot["shares"] for lot in book.lots) * 0.2
            taxes.append(tax_on(*book.sell(t, shares, price[t])))
    rows.append({"method": method, "tax paid during the period": sum(taxes),
                 "unrealised gain left": sum(lot["shares"] * (price[-1] - lot["price"]) for lot in book.lots)})
print(pd.DataFrame(rows).round(0).to_string(index=False))
print(
    "Total gains are the same; the methods differ in *when* they are taxed and at which\n"
    "rate. HIFO realises the smallest gains first and defers the rest. Brokers default\n"
    "to FIFO, and specific identification has to be elected before the trade settles."
)


# ---------------------------------------------------------------------------
header("B. Tax-loss harvesting")
# ---------------------------------------------------------------------------
def harvest_path(seed: int, years: int = 20, threshold: float = 0.10) -> tuple[float, float]:
    r = np.random.default_rng(seed)
    n = TRADING_DAYS * years
    p = 100 * np.exp(np.cumsum(r.normal(0.07 / TRADING_DAYS, 0.20 / np.sqrt(TRADING_DAYS), n)))
    # Buy and hold: one lot, taxed as long-term on liquidation at the end.
    hold = 1e6 * p[-1] / p[0]
    hold_after_tax = hold - LT_RATE * (hold - 1e6)
    # Harvest: whenever the price is 10% below the basis, sell, book the loss, and buy a
    # near-identical substitute (avoiding the wash-sale rule). Losses save tax at the
    # short-term rate against other gains; tax savings are reinvested.
    basis_price, basis_value, shares, buy_day = p[0], 1e6, 1e6 / p[0], 0
    for t in range(1, n):
        if p[t] < (1 - threshold) * basis_price:
            loss = shares * (basis_price - p[t])
            rate = ST_RATE if t - buy_day <= TRADING_DAYS else LT_RATE
            saving = rate * loss
            shares += saving / p[t]
            basis_price, buy_day = p[t], t
            basis_value = shares * p[t]
    value = shares * p[-1]
    after_tax = value - LT_RATE * (value - basis_value)
    return hold_after_tax, after_tax


res = np.array([harvest_path(s) for s in range(300)])
gain = (res[:, 1] / res[:, 0]) ** (1 / 20) - 1
print(f"after-tax terminal wealth, harvesting / buy-and-hold: median {np.median(res[:, 1] / res[:, 0]):.3f}")
print(f"annualised after-tax gain from harvesting: median {np.median(gain):.2%}, 10th-90th pct "
      f"{np.quantile(gain, 0.1):.2%} to {np.quantile(gain, 0.9):.2%}")
print(
    "The gain comes from deferral and from deducting losses at 37% while paying 20%\n"
    "later. It is largest early and for volatile assets (Constantinides 1983;\n"
    "Berkin and Ye 2003)."
)
assert np.median(gain) > 0


# ---------------------------------------------------------------------------
header("C. The `sr_taxlots_2` ledger: reading unfinished code")
# ---------------------------------------------------------------------------
class BranchLedger:
    """compute_tax_lots copied from origin/sr_taxlots_2:finrl/env/accounting/ledger.py:188-240.

    Only the parts needed to call it are reproduced; date parsing uses pd.to_datetime
    without the infer_datetime_format argument that newer pandas removed.
    """

    def __init__(self, tax_threshold_days: int = 365):
        self.d = {"X": {}}
        self.tax_threshold_days = tax_threshold_days

    def date_from_str(self, datestr):
        return pd.to_datetime(datestr).date()

    def log(self, date, shares, price):
        buys, sells = max(0, shares), -min(0, shares)
        self.d["X"][date] = {"buys": buys, "sells": sells, "price": price, "tax_used": 0}
        if sells > 0:
            return self.compute_tax_lots("X", sells, date)

    def compute_tax_lots(self, asset, sells, sell_date):
        a_data = copy(self.d[asset])
        remaining_shares = sells
        long_shares = 0
        long_total_value = 0
        short_shares = 0
        short_total_value = 0
        dates = sorted(a_data)
        i = 0
        while remaining_shares > 0:
            date = dates[i]
            long_term = (
                self.date_from_str(sell_date) - datetime.timedelta(days=self.tax_threshold_days)
            ) > self.date_from_str(date)
            d = a_data[date]
            avail_shares = d["buys"] - d["tax_used"]
            if avail_shares > 0:
                shares_consumed = avail_shares if avail_shares < remaining_shares else remaining_shares
                remaining_shares -= shares_consumed
                if long_term:
                    long_shares += shares_consumed
                    long_total_value += shares_consumed * d["price"]
                else:
                    short_shares += shares_consumed
                    short_total_value += shares_consumed * d["price"]
                a_data[date]["tax_used"] = shares_consumed  # <-- line 224 on the branch
            i += 1
        self.d[asset] = a_data

        def avg(shares, value):
            return 0 if shares == 0 else value / shares

        return {"long_term_shares": long_shares, "long_avg_price": avg(long_shares, long_total_value),
                "short_term_shares": short_shares, "short_avg_price": avg(short_shares, short_total_value)}


ledger = BranchLedger()
ledger.log("2020-01-02", +100, 10.0)
ledger.log("2020-06-01", +100, 20.0)
sales = [("2021-03-01", 60), ("2021-03-02", 60), ("2021-03-03", 60)]
print("Trades: buy 100 @ $10, buy 100 @ $20, then sell 60 three times (FIFO).")
print(f"{'sale':12s} {'branch: avg basis':>18s} {'correct FIFO basis':>19s}")
correct = Lots("FIFO")
correct.buy(0, 100, 10.0)
correct.buy(1, 100, 20.0)
branch_bases, correct_bases = [], []
for k, (date, q) in enumerate(sales):
    out = ledger.log(date, -q, 0.0)
    shares = out["long_term_shares"] + out["short_term_shares"]
    basis = (out["long_term_shares"] * out["long_avg_price"] + out["short_term_shares"] * out["short_avg_price"]) / shares
    st, lt = correct.sell(10 + k, q, 0.0)  # at a price of 0 the "gain" is minus the basis
    right = -(st + lt) / q
    branch_bases.append(basis)
    correct_bases.append(right)
    print(f"{date:12s} {basis:18.2f} {right:19.2f}")
print(
    "\nOn the third sale the branch uses $10 shares again, although only 100 were bought\n"
    "at $10 and 100 have already been sold. `tax_used` is overwritten (=) instead of\n"
    "accumulated (+=), so a partly used lot 'forgets' earlier sales. The fix is one\n"
    "character; finding it needs a test like this one."
)
assert branch_bases[:2] == correct_bases[:2] and branch_bases[2] != correct_bases[2]
print("\nAll Module 09 checks passed.")
