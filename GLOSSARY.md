# Glossary

A reference for the terms, signal outputs, and methods used in this project.

## Signal output (per ticker, as of a date)

- **score** — the final conviction reading, $0$–$100$. A relative ranking across all tickers with congressional buying in the lookback window on a given date. $100$ is the strongest in that day's cross-section. It is a ranking, not a return forecast.
- **n_buys** — total disclosed purchase records for the ticker in the lookback window.
- **cluster** — the number of *distinct* members buying the ticker. Measures how many independent people are agreeing on the name.
- **committee_alignment** — the number of buys made by a member whose committee oversees the ticker's sector.
- **recency** — a recency-weighted count of the buys; more recent disclosures count more.
- **size_vs_networth** — the summed conviction of the buys, where each buy's conviction is its dollar size divided by the buyer's net worth. Controls for the wealth confound.
- **bipartisan** — $1$ when both a Democrat and a Republican bought the ticker, otherwise $0$.
- **`*_n` columns** — each feature scaled $0$–$1$ across the day's cross-section. These normalized values are the actual inputs to the weighted score.

## Methodology

- **cross-sectional normalization** — scaling a feature across all tickers on a single date. A score of $100$ means "top of today's ranking."
- **snapshot** — the signal for one date. `compute(as_of=d)` returns one score per ticker as of date `d`.
- **panel / time series** — a snapshot computed for each date across a range, giving a ticker-by-date grid of scores. Scores do not accumulate across days; each date is an independent, bounded reading, and older activity ages out of the lookback window.
- **lookback window** — the span of disclosures the signal considers (default $30$ days).
- **disclosure (report) date vs transaction date** — the report date is when a trade became public, often weeks after the trade. Signals key on the report date so they never use information that was not yet available.
- **look-ahead bias** — using information in a backtest that would not have been known at the time. Keying on the report date prevents it.
- **feature weight** — the share each feature contributes to the score. Current weights are priors, to be calibrated by the backtest.

## Architecture

- **BaseSignal** — the contract every signal implements: a `compute(as_of)` returning a $0$–$100$ score per ticker.
- **composite score** — a weighted blend of several signal scores into one ranking.
- **enrichment layer** — supplies member net worth, committee membership, and ticker sectors, with synthetic (mock) and live implementations.
- **mock vs live mode** — mock uses synthetic trades and needs no API key; live pulls real Quiver data.

## Data sources

- **Congressional trading**, **Government contracts**, **Lobbying**, **Off-exchange** — Quiver datasets.
- **Committee membership** — @unitedstates/congress-legislators.
- **Net worth** — curated reference table in `reference_data/networth.csv`.

## Backtest terms (notebook 02)

- **forward return** — return over a fixed horizon after a signal date (e.g., $5$, $10$, $21$ trading days).
- **event study** — average forward returns following each signal occurrence.
- **hit rate** — share of signaled trades that were profitable.
- **Sharpe ratio** — risk-adjusted return: average excess return over its volatility.
- **Sortino ratio** — like Sharpe, penalizing only downside volatility.
- **maximum drawdown** — largest peak-to-trough decline over the test period.
- **CAGR** — compound annual growth rate of the strategy.
