"""Synthetic data shaped like the Quiver Quantitative API responses.

Column names and value formats mirror the live `quiverquant` package so the same
normalization works on mock and real data. A few tickers carry heavy, purchase-
skewed activity, and some buys are routed to representatives whose committee aligns
with the traded sector, so the signal's features have structure to detect. Member
names are real, so the live enrichment layer resolves them directly.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

from ..enrichment import mock as menr

_UNIVERSE = ["PLTR", "LMT", "RTX", "AXON", "NOC", "CELH", "MELI", "CAVA",
             "NVDA", "AAPL", "MSFT", "GE", "BA", "CAT", "JPM"]
_HOT = ["PLTR", "LMT", "AXON"]
_RANGES = ["$1,001 - $15,000", "$15,001 - $50,000", "$50,001 - $100,000",
           "$100,001 - $250,000", "$250,001 - $500,000", "$500,001 - $1,000,000"]
_REPS = list(menr.REP_COMMITTEE.keys())
_TITLES = ["CEO", "CFO", "Director", "President", "EVP", "VP Finance", "COO"]
_AGENCIES = ["Department of Defense", "Department of Energy", "NASA",
             "Department of Homeland Security", "Department of Health"]
_ISSUES = ["Defense", "Healthcare", "Taxation", "Energy", "Technology", "Trade"]


def _recent(rng, max_days_ago, min_days_ago=0):
    return date.today() - timedelta(days=int(rng.integers(min_days_ago, max_days_ago)))


def mock_congress_trading(seed=42, n=180, history_days=40):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        if rng.random() < 0.45:
            ticker = rng.choice(_HOT)
            txn = rng.choice(["Purchase", "Sale"], p=[0.85, 0.15])
        else:
            ticker = rng.choice(_UNIVERSE)
            txn = rng.choice(["Purchase", "Sale"], p=[0.55, 0.45])

        sector = menr.TICKER_SECTOR.get(ticker)
        aligned = menr.SECTOR_REPS.get(sector, [])
        rep = rng.choice(aligned) if (aligned and rng.random() < 0.5) else rng.choice(_REPS)

        report = _recent(rng, history_days)
        transaction = report - timedelta(days=int(rng.integers(10, 45)))
        rows.append({
            "Representative": rep,
            "Party": "D" if _REPS.index(rep) % 2 == 0 else "R",
            "House": rng.choice(["Representatives", "Senate"], p=[0.75, 0.25]),
            "Ticker": ticker, "Transaction": txn, "Range": rng.choice(_RANGES),
            "TransactionDate": transaction, "ReportDate": report,
        })
    return pd.DataFrame(rows)


def mock_insiders(seed=43, n=120):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ticker = rng.choice(_HOT) if rng.random() < 0.4 else rng.choice(_UNIVERSE)
        rows.append({
            "Ticker": ticker, "Name": f"Insider {rng.integers(1000, 9999)}",
            "Title": rng.choice(_TITLES), "Date": _recent(rng, 90),
            "TransactionCode": rng.choice(["P", "S"], p=[0.6, 0.4]),
            "Shares": int(rng.integers(500, 50000)),
            "PricePerShare": round(float(rng.uniform(20, 400)), 2),
        })
    return pd.DataFrame(rows)


def mock_gov_contracts(seed=44, n=80, history_days=120):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ticker = rng.choice(_HOT) if rng.random() < 0.5 else rng.choice(_UNIVERSE)
        rows.append({
            "Ticker": ticker, "Date": _recent(rng, history_days),
            "Amount": int(rng.integers(50_000, 500_000_000)),
            "Agency": rng.choice(_AGENCIES), "Description": "Procurement contract award",
        })
    return pd.DataFrame(rows)


def mock_lobbying(seed=45, n=140, history_days=360):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ticker = rng.choice(_UNIVERSE)
        rows.append({
            "Ticker": ticker, "Date": _recent(rng, history_days),
            "Amount": int(rng.integers(10_000, 5_000_000)),
            "Client": f"{ticker} Inc.", "Issue": rng.choice(_ISSUES),
        })
    return pd.DataFrame(rows)


def mock_offexchange(seed=46, history_days=90):
    """Synthetic off-exchange data matching Quiver's live schema.

    Columns: Ticker, Date, OTC_Short, OTC_Total, DPI (= OTC_Short / OTC_Total).
    The live feed is a single daily snapshot per ticker; the mock spans several dates so
    backtests have history to step through. Hot tickers carry a higher DPI (read as more
    off-exchange buying pressure) and heavier off-exchange volume.
    """
    rng = np.random.default_rng(seed)
    rows = []
    steps = max(history_days // 3, 1)
    for ticker in _UNIVERSE:
        base_dpi = 0.50 + (0.07 if ticker in _HOT else 0.0) + rng.uniform(-0.02, 0.02)
        base_vol = int(rng.integers(5_000_000, 60_000_000))
        for k in range(steps):
            d = date.today() - timedelta(days=k * 3)
            dpi = min(0.78, max(0.30, base_dpi + rng.normal(0, 0.02)))
            otc_total = int(base_vol * rng.uniform(0.7, 1.3))
            otc_short = int(otc_total * dpi)
            rows.append({"Ticker": ticker, "Date": d, "OTC_Short": otc_short,
                         "OTC_Total": otc_total, "DPI": round(dpi, 6)})
    return pd.DataFrame(rows)
