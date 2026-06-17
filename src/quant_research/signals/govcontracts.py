"""Government-contract award signal.

Hypothesis: a company winning federal contracts — especially an accelerating flow of
them, across several agencies — has a fundamental tailwind the market may not have
fully priced. The score rewards the level of recent award dollars, their acceleration
over the prior run-rate, the number of distinct wins, and the breadth of awarding
agencies.

Features (each normalized across the cross-section, then weighted):
  award_value    - total award dollars in the lookback window
  acceleration   - recent award dollars above the prior run-rate
  n_awards       - number of distinct awards (repeated wins)
  agency_breadth - number of distinct awarding agencies

Awards key on the announcement date, which is public when it posts, so the signal
carries no look-ahead.
"""
from datetime import date
import pandas as pd

from .base import BaseSignal

DEFAULT_WEIGHTS = {
    "award_value": 0.40,
    "acceleration": 0.30,
    "n_awards": 0.20,
    "agency_breadth": 0.10,
}


class GovContractsSignal(BaseSignal):
    name = "gov_contracts"
    description = "Federal contract-award flow, weighted by acceleration and breadth."

    def __init__(self, client, lookback_days=90, recent_days=30, weights=None):
        self.client = client
        self.lookback_days = lookback_days
        self.recent_days = recent_days
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def compute(self, as_of=None, contracts=None):
        as_of = pd.Timestamp(as_of or date.today())
        start = as_of - pd.Timedelta(days=self.lookback_days)
        recent_start = as_of - pd.Timedelta(days=self.recent_days)

        df = self.client.gov_contracts() if contracts is None else contracts
        win = df[(df.award_date > start) & (df.award_date <= as_of)].copy()
        if win.empty:
            return pd.DataFrame(columns=["score"])

        g = win.groupby("ticker")
        idx = g.size().index
        recent = win[win.award_date > recent_start].groupby("ticker").amount.sum().reindex(idx).fillna(0)
        prior = win[win.award_date <= recent_start].groupby("ticker").amount.sum().reindex(idx).fillna(0)
        prior_days = max(self.lookback_days - self.recent_days, 1)
        prior_rate = prior * (self.recent_days / prior_days)   # put prior on a recent-window footing

        feat = pd.DataFrame({
            "award_value": g.amount.sum(),
            "acceleration": recent - prior_rate,
            "n_awards": g.size(),
            "agency_breadth": g.agency.nunique(),
        })

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
