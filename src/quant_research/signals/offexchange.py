"""Off-exchange (dark-pool) activity signal.

A *confirming* signal, not a directional one. The Quiver off-exchange feed is a daily
snapshot, one row per ticker, reporting off-exchange volume (``OTC_Total``), the short
portion of it (``OTC_Short``), and ``DPI = OTC_Short / OTC_Total`` -- the dark-pool short
ratio. Under the common reading of this metric, off-exchange short volume is the
market-maker side of lit buying (you buy, the dark-pool MM sells short to fill you), so a
higher ratio is interpreted as net accumulation. That interpretation is contested, which
is why this signal carries the lowest weight in the composite and is used to corroborate
rather than to call direction on its own.

Because the feed is a single snapshot with no per-ticker history, the signal is purely
cross-sectional -- there is no trend feature, and it cannot be backtested historically
from the live feed (the mock spans several dates so the engine still has something to
step through).

Features (each normalized across the cross-section, then weighted):
  dpi     - off-exchange short ratio; buying-pressure proxy
  volume  - off-exchange volume (OTC_Total); institutional footprint

A liquidity floor (``min_oe_volume``) drops thinly-traded names whose ratios are
degenerate (e.g. a micro-cap that prints DPI = 1.0 on a single sparse trade).
"""
from datetime import date
import pandas as pd

from .base import BaseSignal

DEFAULT_WEIGHTS = {
    "dpi": 0.65,
    "volume": 0.35,
}


class OffExchangeSignal(BaseSignal):
    name = "off_exchange"
    description = "Off-exchange (dark-pool) buying-pressure proxy; a confirming signal."

    def __init__(self, client, lookback_days=60, min_oe_volume=1_000_000, weights=None):
        self.client = client
        self.lookback_days = lookback_days
        self.min_oe_volume = min_oe_volume
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def compute(self, as_of=None, flow=None):
        as_of = pd.Timestamp(as_of or date.today())
        start = as_of - pd.Timedelta(days=self.lookback_days)

        df = self.client.off_exchange() if flow is None else flow
        win = df[(df.date > start) & (df.date <= as_of)].copy()
        if win.empty:
            return pd.DataFrame(columns=["score"])

        # one row per ticker (the feed is a daily snapshot); take the most recent
        win = win.sort_values("date").groupby("ticker", as_index=False).tail(1)
        win = win[win.oe_volume >= self.min_oe_volume]
        if win.empty:
            return pd.DataFrame(columns=["score"])

        feat = win.set_index("ticker")[["dpi", "oe_volume"]].rename(
            columns={"oe_volume": "volume"})
        norm = pd.DataFrame({f: self._norm(feat[f]) for f in self.weights})
        raw = sum(self.weights[f] * norm[f] for f in self.weights)
        feat["score"] = self._scale_0_100(raw).round(1)
        for f in norm.columns:
            feat[f + "_n"] = norm[f].round(3)
        return feat.sort_values("score", ascending=False)

    @staticmethod
    def _norm(s):
        s = s.astype(float)
        spread = s.max() - s.min()
        return (s - s.min()) / spread if spread > 0 else s * 0.0
