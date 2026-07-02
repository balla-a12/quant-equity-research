# Quant Equity Research: Alternative-Data Conviction Signals

A research pipeline that builds equity-conviction signals from congressional trades, government contracts, lobbying activity, and off-exchange volume, measures their edge with a look-ahead-free backtest, and surfaces the results in a deployable dashboard.

**Stack:** Python · pandas · numpy · scipy · plotly · yfinance · quiverquant · Streamlit

---

## Key Findings

This project tests whether the alternative-data signals behind popular congressional-tracking strategies hold up once market exposure and disclosure timing are accounted for.

- The long-only conviction portfolio's $+10.8\%$ CAGR decomposes to $\beta = 0.91$ ($R^2 = 0.80$) with annualized $\alpha \approx -0.2\%$, attributing the headline return to market exposure while selection $\alpha$ lands near zero.
- A walk-forward, empirical-Bayes member-skill weighting lifts the congressional signal's information coefficient from $0.014$ to $0.017$ (positive in $59\%$ of periods), isolating a small real effect in specific members' trades.
- The government-contracts component measured anti-predictive over the window ($\text{IC} = -0.21$), a result the pipeline surfaces and reweights around.
- Every signal keys on the public disclosure date, keeping the backtest free of look-ahead; the $45$-day disclosure lag caps the achievable short-horizon edge.

**Takeaway:** the signals are marginal as standalone alpha, so the repository's value is a rigorous, look-ahead-free testbed paired with a monitoring dashboard that presents each ranking alongside the evidence that earned it.

---

## What this project answers

Vendors and copytrading products advertise congressional-tracking strategies with headline CAGRs above $30\%$. This repository asks a sharper question: once market exposure and disclosure timing are handled honestly, how much of that return is selection skill? The pipeline ingests four [Quiver Quantitative](https://www.quiverquant.com/) datasets, scores each into a $0$–$100$ conviction measure, blends them, and evaluates the blend on a walk-forward basis.

## Headline findings

- The long-only conviction portfolio returned $+10.8\%$ CAGR over 2022 onward. A regression on the benchmark attributes it to $\beta = 0.91$ (with $R^2 = 0.80$) and an annualized $\alpha \approx -0.2\%$, so the double-digit return reflects market exposure with selection alpha near zero.
- The congressional signal carries a modest, real cross-sectional edge. A dynamic, walk-forward member-skill weighting raises its information coefficient from $0.014$ to $0.017$, positive in $59\%$ of periods.
- Every signal is keyed on the public disclosure date, so the backtest stays free of look-ahead. Congressional trades surface up to $45$ days after execution, which caps the achievable short-horizon edge and shows up directly in the results.
- The government-contracts signal ran anti-predictive over the test window ($\text{IC} = -0.21$). The repository surfaces that finding and reweights around it rather than burying it.

## How it works

### Signals

Each dataset becomes a cross-sectional $0$–$100$ score.

- **Congressional trades** — clustered member purchases, weighted by trade size relative to the member's net worth and by committee-sector alignment.
- **Government contracts** — award value, acceleration, and agency breadth.
- **Lobbying** — spend, acceleration, and issue breadth.
- **Off-exchange volume** — dark-pool short ratio and footprint, used as a present-day confirming signal.

A composite blends the four and rewards names corroborated across datasets, since a security absent from a signal contributes zero to that term.

### Backtest

The engine measures a signal three complementary ways.

- An **event study** reporting the information coefficient and quintile spreads across overlapping weekly rebalances.
- A **market-neutral long-short** curve that isolates $\alpha$ by stripping market $\beta$.
- A **long-only top-quintile** curve with a $\beta / \alpha$ decomposition against an equal-weight benchmark.

Forward returns key on disclosure dates, and the equity curves use non-overlapping holds so the tradeable numbers reflect capital that could actually be deployed.

### Dynamic member-skill weighting

Each member's realized edge is estimated from the excess forward return of their matured buys, then shrunk toward zero with an empirical-Bayes weight $n / (n + k)$ so a thin record earns little influence. The estimate is re-fit at every rebalance date using only trades whose holding window has already closed, which keeps the walk-forward evaluation honest.

## Repository layout

```
src/quant_research/     installable package: ingestion, enrichment, signals, backtest, member_skill
dashboard/              Streamlit app: app.py, panels.py, requirements.txt
reference_data/         member net-worth reference table
tests/                  unit tests for ingestion, enrichment, and the backtest engine
01-08 notebooks         the research narrative, from signal construction to the dashboard
```

| Notebook | Focus |
| --- | --- |
| `01` | Signal construction and the enrichment layer |
| `02` | Backtest engine and the congressional signal |
| `03`-`05` | Government-contract, lobbying, and off-exchange signals |
| `06` | Composite scorer and the honest backtest |
| `07` | Optimizations: score-ranked universe, $\beta/\alpha$ split, high-conviction and dynamic member-skill congress |
| `08` | Conviction dashboard |

## Dashboard

A Streamlit app that shows today's composite ranking with its per-signal breakdown, the trending names in each dataset, the member-skill leaderboard, and the validated backtest evidence with its caveats attached. It runs on bundled mock data with no key, and a Quiver token switches it to live.

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

To deploy on Streamlit Community Cloud, point a new app at this repository with `dashboard/app.py` as the entry point, then add `QUIVER_API_KEY` under Secrets for live data.

## Getting started

```bash
git clone https://github.com/<user>/quant-equity-research.git
cd quant-equity-research
pip install -e .
```

The package runs in mock mode with no credentials, so every notebook and the dashboard work out of the box. For live data, store a Quiver token and reference it through Colab Secrets or the `QUIVER_API_KEY` environment variable.

## Data source

[Quiver Quantitative](https://www.quiverquant.com/) alternative-data API, covering congressional trading, government contracts, lobbying, and off-exchange short volume. The Hobbyist tier covers these four datasets; insider and institutional signals sit on higher tiers and are noted as future work.

## Limitations and honest framing

As standalone systematic alpha, the signals are marginal: information coefficients near $0.01$–$0.02$, positive just over half the time, with a long-short Sharpe below $1$ before costs. The value here is a transparent monitoring and idea-generation tool with the evidence attached, backed by a research process that measured weak and negative signals honestly — including one component that ran backwards — and reweighted on the evidence. The dashboard is best read as a calibrated watchlist that shows where to look.

## Future work

- Add insider (Form 4) and institutional 13F signals on a higher Quiver tier.
- Extend the dynamic member-skill model with a characteristic-based learner for members with thin records.
- Condition the event study on political and market regimes.

---

*Built as a portfolio project for the WorldQuant University MScFE program. Not investment advice.*
