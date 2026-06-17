"""Corporate-lobbying signal.

Hypothesis: a company increasing its lobbying spend — especially an accelerating
spend across many policy issues — is investing to shape the rules that affect it,
which can precede favorable regulatory outcomes the market prices in slowly. The
score blends the level of recent spend, its acceleration over the prior run-rate, the
breadth of issues engaged, and filing intensity.

Features (each normalized across the cross-section, then weighted):
  spend          - total lobbying dollars in the lookback window
  acceleration   - recent spend above the prior run-rate
  issue_breadth  - number of distinct issues lobbied
  n_filings      - number of filings (intensity of engagement)

Filings key on the disclosure (filing) date, public when it posts, so the signal
carries no look-ahead. Lobbying is reported quarterly, so the lookback is longer than
the event-driven signals and acceleration is measured quarter-over-quarter.
"""
from datetime import date
import pandas as pd

from .base import BaseSignal

DEFAULT_WEIGHTS = {
    "spend": 0.40,
    "acceleration": 0.30,
    "issue_breadth": 0.20,
    "n_filings": 0.10,
}


class LobbyingSignal(BaseSignal):
    name = "lobbying"
    description = "Corporate lobbying spend, weighted by acceleration and issue breadth."

    def __init__(self, client, lookback_days=180, recent_days=90, weights=None):
        self.client = client
        self.lookback_days = lookback_days
        self.recent_days = recent_days
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def compute(self, as_of=None, filings=None):
        as_of = pd.Timestamp(as_of or date.today())
        start = as_of - pd.Timedelta(days=self.lookback_days)
        recent_start = as_of - pd.Timedelta(days=self.recent_days)

        df = self.client.lobbying() if filings is None else filings
        win = df[(df.filing_date > start) & (df.filing_date <= as_of)].copy()
        if win.empty:
            return pd.DataFrame(columns=["score"])

        g = win.groupby("ticker")
        idx = g.size().index
        recent = win[win.filing_date > recent_start].groupby("ticker").amount.sum().reindex(idx).fillna(0)
        prior = win[win.filing_date <= recent_start].groupby("ticker").amount.sum().reindex(idx).fillna(0)
        prior_days = max(self.lookback_days - self.recent_days, 1)
        prior_rate = prior * (self.recent_days / prior_days)

        feat = pd.DataFrame({
            "spend": g.amount.sum(),
            "acceleration": recent - prior_rate,
            "issue_breadth": g.issue.nunique(),
            "n_filings": g.size(),
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
