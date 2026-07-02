"""Data layer for the dashboard, kept free of Streamlit so it can be tested directly.

Every function returns plain pandas objects or dicts; the Streamlit app in ``app.py``
renders them. The adopted configuration from notebook 07 drives the numbers: a
score-ranked universe, the dynamic member-skill weighting for the member leaderboard, and
the beta/alpha framing for the backtest-evidence panel.
"""
import numpy as np
import pandas as pd

from quant_research.ingestion.client import QuiverClient
from quant_research.signals.congress import CongressSignal
from quant_research.signals.govcontracts import GovContractsSignal
from quant_research.signals.lobbying import LobbyingSignal
from quant_research.signals.offexchange import OffExchangeSignal
from quant_research.signals.composite import CompositeSignal
from quant_research.signals.member_skill import precompute_buy_returns, member_skill

PARTS = {"congress": 0.40, "gov_contracts": 0.30, "lobbying": 0.20, "off_exchange": 0.10}


def build_client(use_live, token=None, mock_history_days=600):
    return (QuiverClient(token=token, mock_history_days=mock_history_days) if use_live
            else QuiverClient(mock=True, mock_history_days=mock_history_days))


def load_signals(client):
    return {"congress": CongressSignal(client),
            "gov_contracts": GovContractsSignal(client),
            "lobbying": LobbyingSignal(client),
            "off_exchange": OffExchangeSignal(client)}


def composite_ranking(client):
    """Today's blended conviction ranking across all four signals."""
    signals = load_signals(client)
    live = CompositeSignal(signals).compute()
    return live


def trending_by_signal(client, n=8):
    """Top names in each dataset on its own scale, for the monitoring view."""
    signals = load_signals(client)
    out = {}
    for name, sig in signals.items():
        r = sig.compute()
        if len(r):
            out[name] = r[["score"]].head(n).round(1)
    return out


def _synthetic_prices(universe, ranks, start="2022-01-01", seed=7):
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp.today())
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(
            sum(0.0012 * (ranks.get(nm, pd.Series()).get(t, 0.5) - 0.5) for nm in ranks),
            0.010, len(days)))) for t in universe}, index=days)


def member_leaderboard(client, use_live=False, price_n=250, horizon=21, top=8):
    """Walk-forward member-skill multipliers as of today, top and bottom members.

    Needs a price panel to score matured buys. Live mode pulls real prices; mock mode
    generates a synthetic panel tied to the signals so the estimate is well defined.
    """
    signals = load_signals(client)
    trades = client.congress_trades(historical=True)
    ranked = CompositeSignal({k: signals[k] for k in ["congress", "gov_contracts", "lobbying"]}
                             ).compute()
    universe = ranked.head(price_n).index.tolist()

    if use_live:
        from quant_research.backtest.prices import price_history
        prices = price_history(universe, pd.Timestamp("2022-01-01"),
                               pd.Timestamp.today().normalize())
    else:
        KW = {"congress": "trades", "gov_contracts": "contracts", "lobbying": "filings"}
        pf = {"congress": trades, "gov_contracts": client.gov_contracts(),
              "lobbying": client.lobbying()}
        ranks = {nm: signals[nm].compute(**{KW[nm]: pf[nm]})["score"].rank(pct=True)
                 for nm in KW}
        prices = _synthetic_prices(universe, ranks)

    br = precompute_buy_returns(trades, prices, horizon=horizon)
    mult = member_skill(br, pd.Timestamp.today(), prior_strength=20.0)
    if not len(mult):
        return None, 0, 0
    mult = mult.sort_values(ascending=False)
    board = pd.concat([mult.head(top), mult.tail(top)]).round(2).to_frame("skill multiplier")
    return board, len(mult), len(br)


def backtest_evidence():
    """Validated figures from notebook 07, shown as documented results with caveats."""
    variants = pd.DataFrame([
        {"congress variant": "base", "IC (21d)": 0.014, "positive %": 57, "long-short %/mo": 0.260},
        {"congress variant": "high-conviction", "IC (21d)": 0.014, "positive %": 57, "long-short %/mo": 0.238},
        {"congress variant": "dynamic member-skill", "IC (21d)": 0.017, "positive %": 59, "long-short %/mo": 0.369},
    ]).set_index("congress variant")
    decomposition = {"long_only_cagr": 0.108, "beta": 0.91, "alpha_annual": -0.002, "r_squared": 0.80}
    caveats = [
        "Signals are keyed on the disclosure date, so the edge is measured on information "
        "already public — no look-ahead, and much of the short-horizon timing decays by then.",
        "The long-only CAGR is roughly 0.91-beta market exposure; the selection alpha is near zero.",
        "Event-study periods overlap, so the effective independent sample is far below the raw count.",
        "This is a monitoring and idea-generation view, calibrated to the evidence, rather than a buy list.",
    ]
    return variants, decomposition, caveats
