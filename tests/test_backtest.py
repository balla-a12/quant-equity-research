"""The engine must detect edge when present and report null on random scores."""
import numpy as np
import pandas as pd
from quant_research.backtest.engine import event_study


def _prices(seed=0, n_tickers=40, n_days=400):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    steps = rng.normal(0, 0.01, size=(n_days, n_tickers))
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, 0)), index=dates, columns=cols), rng


def test_engine_detects_edge():
    prices, rng = _prices()
    rebal = pd.bdate_range("2021-02-01", "2022-06-01", freq="W-FRI")
    H = 21

    def score_edge(d):
        pos = prices.index.searchsorted(pd.Timestamp(d))
        if pos + H >= len(prices.index):
            return None
        fwd = prices.iloc[pos + H] / prices.iloc[pos] - 1
        return pd.Series(fwd + rng.normal(0, 0.02, len(fwd)), index=prices.columns)

    s = event_study(score_edge, prices, rebal, horizon=H)["summary"]
    assert s["mean_ic"] > 0.3
    assert s["mean_long_short"] > 0


def test_engine_reports_null_on_noise():
    prices, rng = _prices(seed=1)
    rebal = pd.bdate_range("2021-02-01", "2022-06-01", freq="W-FRI")
    s = event_study(lambda d: pd.Series(rng.normal(size=prices.shape[1]),
                                        index=prices.columns),
                    prices, rebal, horizon=21)["summary"]
    assert abs(s["mean_ic"]) < 0.15
