"""Event-study backtester for a cross-sectional signal.

On each rebalance date the engine scores the universe, sorts names into buckets by
score, and measures their forward return over a fixed horizon. Aggregated across
dates, this answers the core question: do higher-scoring names earn higher forward
returns? It reports three complementary views:

  - bucket means: average forward return by score bucket (monotonic is the goal)
  - information coefficient (IC): rank correlation of score with forward return,
    averaged across dates, with the share of dates where it is positive
  - a long-top-bucket equity curve with standard risk metrics

The engine takes a `score_fn(date) -> Series[ticker -> score]`, so it is agnostic to
which signal it is testing.
"""
import numpy as np
import pandas as pd

from .prices import forward_returns


def _spearman(a, b):
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return a.rank().corr(b.rank())


def event_study(score_fn, prices, rebalance_dates, horizon=21, n_buckets=5):
    bucket_rows, ic_rows, ls_rows = [], [], []
    for d in rebalance_dates:
        scores = score_fn(d)
        if scores is None or len(scores) < n_buckets:
            continue
        fwd = forward_returns(prices, d, horizon)
        df = pd.concat([scores.rename("score"), fwd.rename("fwd")], axis=1).dropna()
        if len(df) < n_buckets:
            continue
        df["bucket"] = pd.qcut(df["score"].rank(method="first"), n_buckets, labels=False)
        means = df.groupby("bucket")["fwd"].mean()
        for b, m in means.items():
            bucket_rows.append({"date": d, "bucket": int(b), "fwd": m})
        ls_rows.append({"date": d, "long_short": means.get(n_buckets - 1) - means.get(0)})
        ic_rows.append({"date": d, "ic": _spearman(df["score"], df["fwd"])})

    buckets = pd.DataFrame(bucket_rows)
    ls = pd.DataFrame(ls_rows)
    ic = pd.DataFrame(ic_rows).dropna()

    summary = {
        "n_periods": len(ls),
        "mean_ic": ic["ic"].mean() if len(ic) else np.nan,
        "ic_positive_share": (ic["ic"] > 0).mean() if len(ic) else np.nan,
        "mean_long_short": ls["long_short"].mean() if len(ls) else np.nan,
        "ls_positive_share": (ls["long_short"] > 0).mean() if len(ls) else np.nan,
    }
    bucket_means = (buckets.groupby("bucket")["fwd"].mean()
                    if len(buckets) else pd.Series(dtype=float))
    return {"summary": summary, "bucket_means": bucket_means,
            "long_short": ls, "ic": ic}


def feature_buckets(score_fn_full, prices, rebalance_dates, feature, horizon=21):
    """Forward return split by whether a 0/1 feature is present (e.g. committee_alignment>0)."""
    on, off = [], []
    for d in rebalance_dates:
        panel = score_fn_full(d)              # full feature DataFrame indexed by ticker
        if panel is None or panel.empty:
            continue
        fwd = forward_returns(prices, d, horizon)
        df = panel.join(fwd.rename("fwd")).dropna(subset=["fwd"])
        if df.empty:
            continue
        flag = df[feature] > 0
        on.extend(df.loc[flag, "fwd"].tolist())
        off.extend(df.loc[~flag, "fwd"].tolist())
    return {"with_feature_mean": np.mean(on) if on else np.nan, "with_n": len(on),
            "without_feature_mean": np.mean(off) if off else np.nan, "without_n": len(off)}


def long_top_curve(score_fn, prices, rebalance_dates, horizon=21, top_frac=0.2):
    """Equity curve of equal-weighting the top score fraction on a non-overlapping
    schedule. Entries are kept at least `horizon` trading days apart, measured against
    the price index, so the cadence is correct whatever the rebalance-date frequency."""
    idx = prices.index
    rets, last_pos = [], -horizon
    for d in rebalance_dates:
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos - last_pos < horizon:          # keep holding periods from overlapping
            continue
        scores = score_fn(d)
        if scores is None or len(scores) == 0:
            continue
        fwd = forward_returns(prices, d, horizon)
        df = pd.concat([scores.rename("score"), fwd.rename("fwd")], axis=1).dropna()
        if df.empty:
            continue
        k = max(1, int(round(len(df) * top_frac)))
        rets.append(df.nlargest(k, "score")["fwd"].mean())
        last_pos = pos
    rets = pd.Series(rets)
    equity = (1 + rets).cumprod()
    return equity, rets


def long_short_curve(score_fn, prices, rebalance_dates, horizon=21, top_frac=0.2):
    """Market-neutral equity curve: long the top score fraction, short the bottom,
    on a non-overlapping schedule. Isolates the signal from market beta."""
    idx = prices.index
    rets, last_pos = [], -horizon
    for d in rebalance_dates:
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos - last_pos < horizon:
            continue
        scores = score_fn(d)
        if scores is None or len(scores) == 0:
            continue
        df = pd.concat([scores.rename("score"),
                        forward_returns(prices, d, horizon).rename("fwd")], axis=1).dropna()
        if len(df) < 2:
            continue
        k = max(1, int(round(len(df) * top_frac)))
        rets.append(df.nlargest(k, "score")["fwd"].mean() - df.nsmallest(k, "score")["fwd"].mean())
        last_pos = pos
    rets = pd.Series(rets)
    return (1 + rets).cumprod(), rets


def benchmark_curve(prices, rebalance_dates, horizon=21):
    """Equal-weight all available names each non-overlapping period: a universe
    buy-and-hold proxy for separating signal alpha from market beta."""
    idx = prices.index
    rets, last_pos = [], -horizon
    for d in rebalance_dates:
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos - last_pos < horizon:
            continue
        fwd = forward_returns(prices, d, horizon).dropna()
        if fwd.empty:
            continue
        rets.append(fwd.mean())
        last_pos = pos
    rets = pd.Series(rets)
    return (1 + rets).cumprod(), rets


def metrics(rets, horizon=21):
    """Annualized risk/return metrics from a series of per-period returns."""
    if len(rets) == 0:
        return {"cagr": float("nan"), "sharpe": float("nan"), "sortino": float("nan"),
                "max_drawdown": float("nan"), "hit_rate": float("nan"), "n_trades": 0}
    per_year = 252 / horizon
    growth = (1 + rets).prod()
    cagr = growth ** (per_year / len(rets)) - 1
    vol = rets.std(ddof=1)
    sharpe = (rets.mean() / vol * np.sqrt(per_year)) if vol > 0 else np.nan
    downside = rets[rets < 0].std(ddof=1)
    sortino = (rets.mean() / downside * np.sqrt(per_year)) if downside and downside > 0 else np.nan
    equity = (1 + rets).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_dd, "hit_rate": (rets > 0).mean(), "n_trades": len(rets)}
