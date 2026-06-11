"""Synthetic data shaped like the Quiver Quantitative API responses.

This lets the entire pipeline run with no API key. The column names and value
formats mirror what the live `quiverquant` package returns, so the same
normalization code works whether data is mock or real.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

# A small universe. A few tickers are deliberately "hot" (heavy clustered
# congressional buying) so that the signals we build in Module 3 have real
# structure to detect rather than pure noise.
_UNIVERSE = ["PLTR", "LMT", "RTX", "AXON", "NOC", "CELH", "MELI", "CAVA",
             "NVDA", "AAPL", "MSFT", "GE", "BA", "CAT", "JPM"]
_HOT = ["PLTR", "LMT", "AXON"]

_RANGES = ["$1,001 - $15,000", "$15,001 - $50,000", "$50,001 - $100,000",
           "$100,001 - $250,000", "$250,001 - $500,000", "$500,001 - $1,000,000"]
_REPS = ["Nancy Pelosi", "Dan Crenshaw", "Josh Gottheimer", "Marjorie Taylor Greene",
         "Ro Khanna", "Tommy Tuberville", "Michael McCaul", "Virginia Foxx",
         "Earl Blumenauer", "Kathy Manning"]
_TITLES = ["CEO", "CFO", "Director", "President", "EVP", "VP Finance", "COO"]
_AGENCIES = ["Department of Defense", "Department of Energy", "NASA",
             "Department of Homeland Security", "Department of Health"]
_ISSUES = ["Defense", "Healthcare", "Taxation", "Energy", "Technology", "Trade"]


def _recent_date(rng, max_days_ago=90):
    return date.today() - timedelta(days=int(rng.integers(0, max_days_ago)))


def mock_congress_trading(seed=42, n=160):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        # Hot tickers are over-sampled and skew toward purchases
        if rng.random() < 0.45:
            ticker = rng.choice(_HOT)
            txn = rng.choice(["Purchase", "Sale"], p=[0.85, 0.15])
        else:
            ticker = rng.choice(_UNIVERSE)
            txn = rng.choice(["Purchase", "Sale"], p=[0.55, 0.45])
        rows.append({
            "Representative": rng.choice(_REPS),
            "Party": rng.choice(["D", "R", "I"], p=[0.47, 0.47, 0.06]),
            "House": rng.choice(["Representatives", "Senate"], p=[0.75, 0.25]),
            "Ticker": ticker,
            "Transaction": txn,
            "Range": rng.choice(_RANGES),
            "TransactionDate": _recent_date(rng),
        })
    return pd.DataFrame(rows)


def mock_insiders(seed=43, n=120):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ticker = rng.choice(_HOT) if rng.random() < 0.4 else rng.choice(_UNIVERSE)
        code = rng.choice(["P", "S"], p=[0.6, 0.4])
        rows.append({
            "Ticker": ticker,
            "Name": f"Insider {rng.integers(1000, 9999)}",
            "Title": rng.choice(_TITLES),
            "Date": _recent_date(rng),
            "TransactionCode": code,
            "Shares": int(rng.integers(500, 50000)),
            "PricePerShare": round(float(rng.uniform(20, 400)), 2),
        })
    return pd.DataFrame(rows)


def mock_gov_contracts(seed=44, n=80):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ticker = rng.choice(_HOT) if rng.random() < 0.5 else rng.choice(_UNIVERSE)
        rows.append({
            "Ticker": ticker,
            "Date": _recent_date(rng),
            "Amount": int(rng.integers(50_000, 500_000_000)),
            "Agency": rng.choice(_AGENCIES),
            "Description": "Procurement contract award",
        })
    return pd.DataFrame(rows)


def mock_lobbying(seed=45, n=70):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ticker = rng.choice(_UNIVERSE)
        rows.append({
            "Ticker": ticker,
            "Date": _recent_date(rng),
            "Amount": int(rng.integers(10_000, 5_000_000)),
            "Client": f"{ticker} Inc.",
            "Issue": rng.choice(_ISSUES),
        })
    return pd.DataFrame(rows)
