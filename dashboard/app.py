"""Alternative-data conviction dashboard.

A monitoring and idea-generation view over the congressional, government-contract,
lobbying, and off-exchange signals, with the backtest evidence attached so every number
carries its own calibration. Runs on mock data with no key; set a Quiver token to go live.

    streamlit run dashboard/app.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                              # dashboard/ for panels
sys.path.insert(0, os.path.join(_HERE, "..", "src"))   # repo src for quant_research

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import panels

st.set_page_config(page_title="Alt-Data Conviction", layout="wide")


def _token():
    try:
        return st.secrets.get("QUIVER_API_KEY", os.environ.get("QUIVER_API_KEY"))
    except Exception:
        return os.environ.get("QUIVER_API_KEY")


@st.cache_data(show_spinner="Loading signals...")
def get_ranking(use_live):
    return panels.composite_ranking(panels.build_client(use_live, _token()))


@st.cache_data(show_spinner="Scanning datasets...")
def get_trending(use_live):
    return panels.trending_by_signal(panels.build_client(use_live, _token()))


@st.cache_data(show_spinner="Scoring member skill...")
def get_board(use_live):
    return panels.member_leaderboard(panels.build_client(use_live, _token()), use_live=use_live)


# ---- sidebar ----------------------------------------------------------------
st.sidebar.title("Alt-Data Conviction")
live = st.sidebar.toggle("Use live Quiver data", value=False,
                         help="Off uses bundled mock data. On requires a Quiver token in "
                              "Streamlit secrets or the QUIVER_API_KEY environment variable.")
if live and not _token():
    st.sidebar.warning("No token found; staying on mock data.")
    live = False
st.sidebar.caption("Mode: " + ("live" if live else "mock"))
st.sidebar.markdown("---")
st.sidebar.caption("Signals: congressional trades, government contracts, lobbying, "
                   "off-exchange volume. Blended into a 0-100 conviction score.")

# ---- header -----------------------------------------------------------------
st.title("Alternative-Data Conviction Dashboard")
st.markdown(
    "A research and idea-generation view over four alternative-data signals. Scores rank "
    "names by corroborated activity, and the backtest evidence below states how much weight "
    "each ranking has earned. Read it as a calibrated watchlist that surfaces where to look, "
    "with the honest measurement kept in view.")

ranking = get_ranking(live)
trending = get_trending(live)
board, n_members, n_buys = get_board(live)
variants, decomp, caveats = panels.backtest_evidence()

# ---- composite ranking ------------------------------------------------------
st.header("Composite conviction ranking")
left, right = st.columns([3, 2])
with left:
    cols = ["congress", "gov_contracts", "lobbying", "off_exchange", "n_signals", "score"]
    st.dataframe(ranking[cols].head(15).round(1), width='stretch')
with right:
    top = ranking.head(12).iloc[::-1]
    fig = go.Figure()
    for name, w in panels.PARTS.items():
        fig.add_bar(y=top.index, x=top[name].fillna(0) * w, name=name, orientation="h")
    fig.update_layout(barmode="stack", height=420, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="weighted contribution", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width='stretch')
st.caption("A tall bar built from several colors is a name corroborated across datasets. "
           "Absent signals contribute zero, so the score rewards breadth.")

# ---- trending ---------------------------------------------------------------
st.header("Trending by dataset")
cc = st.columns(len(trending))
for col, (name, tbl) in zip(cc, trending.items()):
    col.subheader(name.replace("_", " "))
    col.dataframe(tbl, width='stretch')

# ---- member skill -----------------------------------------------------------
st.header("Member-skill leaderboard")
if board is not None:
    st.caption(f"Walk-forward empirical-Bayes estimate over {n_members} members and "
               f"{n_buys:,} matured buys. A multiplier above one up-weights that member's "
               "trades; the estimate uses only buys whose window closed as of today.")
    st.dataframe(board, width='stretch')
else:
    st.info("Not enough matured trades to score members in this mode.")

# ---- backtest evidence ------------------------------------------------------
st.header("Backtest evidence")
st.markdown("Validated on the score-ranked universe over 2022 onward, weekly rebalances, "
            "walk-forward. The dynamic member-skill variant is the adopted configuration.")
st.dataframe(variants, width='stretch')

m1, m2, m3 = st.columns(3)
m1.metric("Long-only CAGR", f"{decomp['long_only_cagr']*100:+.1f}%")
m2.metric("Market beta", f"{decomp['beta']:.2f}", help="Fraction of the return that is market exposure")
m3.metric("Annualized alpha", f"{decomp['alpha_annual']*100:+.1f}%",
          help=f"Residual after beta; R^2 {decomp['r_squared']:.2f}")
st.markdown("The long-only headline compounds mainly through market beta near one; the "
            "selection alpha after removing it is close to zero. That is the honest reading "
            "of the double-digit CAGR that vendor trackers advertise.")

with st.expander("Caveats worth keeping in view"):
    for c in caveats:
        st.markdown("- " + c)

st.markdown("---")
st.caption("Built on the quant-equity-research package. Signals are keyed on disclosure "
           "dates to stay free of look-ahead. Not investment advice.")
