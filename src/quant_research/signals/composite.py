"""Composite signal: blend the individual signals into one conviction ranking.

Each component signal produces a cross-sectional 0-100 score. The composite takes a
weighted sum of those scores, where a ticker absent from a signal contributes 0 to that
term. That choice is deliberate: it rewards corroboration. A name surfacing in congress
buys *and* government contracts accumulates more weighted points than one appearing in a
single signal, which is exactly the conviction-stacking the project is testing.

With weights that sum to 1 and each component on 0-100, the composite score is itself on
0-100, read as conviction points out of a possible 100 (a name topping every component
at once). In practice no name tops all four, so the realized top scores sit well below
100, and ``n_signals`` shows how many components a name actually appears in.

For the historical backtest, only the event signals carry usable history, so the
composite is built from those; off-exchange is a present-day confirming layer that joins
the live ranking but stays out of the backtest.
"""
from datetime import date
import pandas as pd

DEFAULT_WEIGHTS = {
    "congress": 0.40,
    "gov_contracts": 0.30,
    "lobbying": 0.20,
    "off_exchange": 0.10,
}

# maps a component name to the keyword its compute() uses for prefetched data
_DATA_KWARG = {
    "congress": "trades",
    "gov_contracts": "contracts",
    "lobbying": "filings",
    "off_exchange": "flow",
}


class CompositeSignal:
    name = "composite"
    description = "Weighted blend of the component signals into one conviction score."

    def __init__(self, signals, weights=None):
        """signals: dict of {component_name: signal_instance}.

        Weights default to the project weights restricted to the components supplied, so
        a three-signal backtest composite and a four-signal live composite share one class.
        """
        self.signals = signals
        if weights is None:
            weights = {k: v for k, v in DEFAULT_WEIGHTS.items() if k in signals}
        self.weights = weights

    def compute(self, as_of=None, prefetch=None):
        as_of = pd.Timestamp(as_of or date.today())
        cols = {}
        for nm, sig in self.signals.items():
            kw = {}
            if prefetch and nm in prefetch and _DATA_KWARG.get(nm):
                kw[_DATA_KWARG[nm]] = prefetch[nm]
            res = sig.compute(as_of, **kw)
            cols[nm] = res["score"] if len(res) else pd.Series(dtype=float)

        wide = pd.DataFrame(cols)
        for k in self.weights:
            if k not in wide.columns:
                wide[k] = pd.NA

        w = pd.Series(self.weights)
        present = wide[w.index].notna()
        filled = wide[w.index].fillna(0.0)
        score = filled.mul(w, axis=1).sum(axis=1)

        out = wide.copy()
        out["n_signals"] = present.sum(axis=1).astype(int)
        out["score"] = score.round(1)
        return out.sort_values("score", ascending=False)

    def score_fn(self, prefetch=None):
        """Return a ``score_fn(as_of) -> Series`` for the backtest engine."""
        def _fn(as_of):
            res = self.compute(as_of, prefetch=prefetch)
            return res["score"] if len(res) else None
        return _fn
