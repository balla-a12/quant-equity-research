# Quiver Signals

A quantitative signal-research system built on alternative data. The pipeline ingests
congressional trades, insider transactions, government contracts, and lobbying activity
from the [Quiver Quantitative API](https://api.quiverquant.com), converts them into
normalized signal scores, backtests those signals against historical prices, and surfaces
ranked trade candidates in an interactive dashboard.

## Why this exists

Most quant signals are extracted from data every fund already owns, so the edge is thin.
This project instead mines *alternative* data sources that can correlate with future returns
before the broader market prices them in, and combines several independent sources into a
single weighted conviction score.

## Architecture

| Layer | Module | What it does |
|---|---|---|
| Ingestion | `src/ingestion` | Pull and normalize Quiver datasets |
| Signals | `src/signals` | Convert raw data into 0-100 scores |
| Backtest | `src/backtest` | Measure historical predictive edge |
| Dashboard | dashboard app | Visualize trending signals and candidates |

## Running it

The project ships with a **mock-data mode** so the full pipeline runs with no API key:

```bash
pip install -r requirements.txt
```

Open the notebooks in `notebooks/` in order, starting with `01_foundations.ipynb`.
To switch from mock data to live data, add a Quiver API token (see Module 2).

## Status

Built as a learning project, in public, one module at a time. See `notebooks/` for the
full narrative.

## Disclaimer

For research and educational purposes only. Nothing here is financial advice.
