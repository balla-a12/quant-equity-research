"""Dynamic member-skill weighting for the congressional signal.

A static high-conviction tilt asks *which trades look informed by their attributes*.
This module asks a complementary, data-driven question: *which members have actually
picked winners*, and it lets that estimate evolve as new disclosures mature.

The estimate is built to avoid the two traps that flatter naive member rankings:

  look-ahead  - a member's skill at a rebalance date uses only buys whose holding
                window has already closed on or before that date, so nothing from the
                future leaks in. Wiring this through ``event_study`` gives an honest
                walk-forward evaluation.
  small samples - most members disclose few trades, so a couple of lucky picks would
                dominate a raw average. Each member's edge is shrunk toward zero with
                an empirical-Bayes weight n / (n + k), so a member earns influence only
                as their track record accumulates.

A member's raw edge is the average *excess* forward return of their buys, where excess
is measured against the mean forward return of all congressional buys disclosed the
same month. That removes the market move and isolates selection.
"""
import numpy as np
import pandas as pd


def precompute_buy_returns(trades, prices, horizon=21):
    """One-pass forward returns for every congressional buy that can be priced.

    Returns a frame with the member, ticker, disclosure date, the date the holding
    window closes (``exit_date``), the realized forward return, and its month-demeaned
    excess. Computed once and reused across rebalance dates.
    """
    buys = trades[trades.transaction_type == "Purchase"].copy()
    buys = buys[buys.ticker.isin(prices.columns)]
    if buys.empty:
        return pd.DataFrame(columns=["representative", "ticker", "report_date",
                                     "exit_date", "fwd_ret", "period", "excess"])
    idx = prices.index
    recs = []
    for tkr, grp in buys.groupby("ticker"):
        px = prices[tkr].to_numpy()
        pos = idx.searchsorted(pd.to_datetime(grp.report_date.values))
        for rep, rd, entry in zip(grp.representative, grp.report_date, pos):
            exit_ = entry + horizon
            if entry < len(px) and exit_ < len(px):
                pe, px_e = px[entry], px[exit_]
                if pe and pe > 0 and not np.isnan(pe) and not np.isnan(px_e):
                    recs.append((rep, tkr, pd.Timestamp(rd), idx[exit_], px_e / pe - 1.0))
    out = pd.DataFrame(recs, columns=["representative", "ticker", "report_date",
                                      "exit_date", "fwd_ret"])
    if not out.empty:
        out["period"] = out.report_date.dt.to_period("M")
    return out


def member_skill(buy_rets, as_of, prior_strength=20.0):
    """Empirical-Bayes member multipliers using only buys matured by ``as_of``.

    Returns a Series mapping member -> multiplier (centered near 1.0). Members with no
    matured history are absent, and the signal treats them as neutral.
    """
    if buy_rets is None or len(buy_rets) == 0:
        return pd.Series(dtype=float)
    matured = buy_rets[buy_rets.exit_date <= pd.Timestamp(as_of)].copy()
    if matured.empty:
        return pd.Series(dtype=float)
    matured["excess"] = matured.fwd_ret - matured.groupby("period").fwd_ret.transform("mean")
    grp = matured.groupby("representative").excess
    n, raw = grp.count(), grp.mean()
    shrunk = (n * raw) / (n + prior_strength)        # shrink toward zero edge
    sd = shrunk.std(ddof=0)
    z = (shrunk - shrunk.mean()) / sd if sd and sd > 0 else shrunk * 0.0
    return np.exp(0.5 * z.clip(-2, 2))               # smooth, positive, ~[0.37, 2.72]


def dynamic_congress_score_fn(signal, trades, prices, horizon=21, prior_strength=20.0):
    """A score_fn(as_of) for the backtest engine that re-estimates member skill
    walk-forward at each date and feeds it to the congressional signal as member
    weights. Forward returns are precomputed once for speed.
    """
    buy_rets = precompute_buy_returns(trades, prices, horizon=horizon)

    def score_fn(as_of):
        mult = member_skill(buy_rets, as_of, prior_strength=prior_strength)
        r = signal.compute(as_of, trades=trades,
                           member_weights=(mult if len(mult) else None))
        return r["score"] if len(r) else None

    return score_fn
