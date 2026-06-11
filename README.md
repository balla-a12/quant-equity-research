# Quant Equity Research — Alternative Data Signals

An equity research framework that constructs trading signals from alternative data,
tests them for predictive edge, and translates the survivors into ranked trade
candidates.

## Overview

Traditional quant signals are extracted from data every fund already owns, which
thins their edge. This project instead builds signals from four alternative-data
sources — congressional trading disclosures, corporate insider transactions,
federal contract awards, and lobbying activity — sourced from the
[Quiver Quantitative](https://www.quiverquant.com) API.

The central thesis: a single alternative-data source is noisy, yet independent
sources that converge on the same security carry information the broader market may
not have fully priced. The framework quantifies that convergence as a weighted
composite score on a 0–100 scale and validates it with an event-driven backtest
before any signal is treated as actionable.

## Approach

| Stage | Module | Responsibility |
|---|---|---|
| Ingestion | `src/ingestion` | Normalize four Quiver datasets into one internal schema; persist to SQLite with idempotent writes |
| Signals | `src/signals` | Convert each dataset into a normalized conviction score; blend into a weighted composite |
| Backtest | `src/backtest` | Measure forward-return edge with Sharpe, Sortino, max drawdown, and hit rate |
| Classification | `src/classify` | Map composite score and implied-volatility rank to a trade structure |

## Design notes

- **Normalization at the boundary.** Each dataset arrives with its own column names
  and encodings (a dollar-range string for congressional trades, single-letter
  transaction codes for insiders). The ingestion client resolves these into one
  internal schema, so every downstream component depends on a single stable
  contract rather than on vendor-specific shapes.

- **Reproducible without credentials.** A mock data source mirrors the live API's
  structure, so the full pipeline runs end to end with no API key. The repository is
  therefore runnable by any reviewer, and the path from mock to live is a single
  constructor argument.

- **Idempotent ingestion.** Row identity is a content hash, so re-running ingestion
  never duplicates filings. Repeated pulls converge to the same stored state.

- **No look-ahead.** Signals computed for date *t* use only data available on or
  before *t*. This discipline is what separates a credible backtest from an
  overfit one.

## Repository layout

```
quant-equity-research/
├── pyproject.toml          # package metadata and dependencies
├── src/
│   └── quant_research/      # the installable package
│       ├── ingestion/      # client, normalization, storage
│       ├── signals/        # conviction scoring and composite blend
│       ├── backtest/       # event-driven engine and risk metrics
│       └── classify/       # trade-structure classification
├── notebooks/
│   ├── 01_signal_construction.ipynb
│   ├── 02_backtest_results.ipynb
│   └── 03_dashboard.ipynb
├── tests/
└── README.md
```

## Running it

Install the package in editable mode. This pulls the dependencies and makes
`quant_research` importable from anywhere, including the notebooks:

```bash
pip install -e .
```

The pipeline defaults to mock data, so the notebooks in `notebooks/` run
immediately with no API key.

To run on live data, supply a Quiver API token to the ingestion client:

```python
from quant_research.ingestion.client import QuiverClient
client = QuiverClient(token="YOUR_QUIVER_TOKEN")   # omit token for mock mode
```

## Research notebooks

- **`01_signal_construction.ipynb`** — builds the alternative-data signals, examines
  their distribution across the universe, and explains the scoring methodology.
- **`02_backtest_results.ipynb`** — tests whether the composite signal predicts
  forward returns, with equity curves and risk metrics benchmarked against the
  S&P 500.

## Disclaimer

For research and educational purposes only. Nothing here constitutes investment
advice.
