"""Congressional cluster-buy signal.

The score rewards securities that several members of Congress are buying at once,
with each buy weighted by how informed it is likely to be. Features encode testable
hypotheses; the backtest layer calibrates their weights.

Features (each normalized across the cross-section, then weighted):
  cluster              - distinct members buying the same name
  size_vs_networth     - trade size relative to the member's net worth (conviction
                         relative to means, controlling for the wealth confound)
  committee_alignment  - buys by members whose committee oversees the ticker's sector
  recency              - disclosures weighted toward the present
  bipartisan           - a bonus when both parties are buying the same name

Member net worth and committee membership come from the enrichment layer, which
resolves a real implementation on live data and a synthetic one in mock mode.

Signals are keyed on the DISCLOSURE (report) date, never the trade date, because a
trade is only actionable once it is public — which is what keeps the downstream
backtest free of look-ahead.
"""
from datetime import date
import numpy as np
import pandas as pd

from .base import BaseSignal
from ..enrichment.mock import MockEnrichment
from ..enrichment.live import LiveEnrichment

DEFAULT_WEIGHTS = {
    "cluster": 0.30,
    "size_vs_networth": 0.25,
    "committee_alignment": 0.25,
    "recency": 0.15,
    "bipartisan": 0.05,
}


class CongressSignal(BaseSignal):
    name = "congress"
    description = ("Clustered congressional purchases, weighted by conviction "
                   "relative to net worth and by committee-sector alignment.")

    def __init__(self, client, lookback_days=30, weights=None, enrichment=None):
        self.client = client
        self.lookback_days = lookback_days
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.enrichment = enrichment or (
            MockEnrichment() if getattr(client, "mock", False) else LiveEnrichment())

    def compute(self, as_of=None):
        as_of = pd.Timestamp(as_of or date.today())
        start = as_of - pd.Timedelta(days=self.lookback_days)

        trades = self.client.congress_trades()
        buys = trades[(trades.transaction_type == "Purchase")
                      & (trades.report_date > start)
                      & (trades.report_date <= as_of)].copy()
        if buys.empty:
            return pd.DataFrame(columns=["score"])

        buys["mid_amount"] = (buys.amount_min + buys.amount_max) / 2.0

        nw = buys.representative.map(self.enrichment.net_worth)
        if nw.notna().any():
            # conviction relative to means; unknown members fall back to the median
            buys["conviction"] = buys.mid_amount / nw.fillna(nw.dropna().median())
        else:
            buys["conviction"] = buys.mid_amount      # no net worth available -> raw size

        buys["days_ago"] = (as_of - buys.report_date).dt.days
        buys["recency_w"] = np.select(
            [buys.days_ago <= 7, buys.days_ago <= 14], [1.0, 0.6], default=0.3)
        buys["aligned"] = [int(self.enrichment.is_aligned(r, t))
                           for r, t in zip(buys.representative, buys.ticker)]

        g = buys.groupby("ticker")
        feat = pd.DataFrame({
            "n_buys": g.size(),
            "cluster": g.representative.nunique(),
            "size_vs_networth": g.conviction.sum(),
            "committee_alignment": g.aligned.sum(),
            "recency": g.recency_w.sum(),
            "bipartisan": g.party.apply(lambda p: int({"D", "R"}.issubset(set(p)))),
        })

        norm = {}
        for f in ["cluster", "size_vs_networth", "committee_alignment", "recency"]:
            col = feat[f].astype(float)
            spread = col.max() - col.min()
            norm[f] = (col - col.min()) / spread if spread > 0 else col * 0.0
        norm["bipartisan"] = feat["bipartisan"].astype(float)
        norm = pd.DataFrame(norm)

        raw = sum(self.weights[f] * norm[f] for f in self.weights)
        feat["score"] = self._scale_0_100(raw).round(1)
        for f in norm.columns:
            feat[f + "_n"] = norm[f].round(3)
        return feat.sort_values("score", ascending=False)
