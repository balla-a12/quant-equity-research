"""Price data and forward returns.

The price layer is kept separate from the backtest engine: it fetches adjusted
close prices (via yfinance) into a wide DataFrame, and the engine consumes that
DataFrame. This separation lets the engine's logic be tested with synthetic prices
that have a known relationship to the signal.
"""
import pandas as pd


def price_history(tickers, start, end):
    """Wide DataFrame of adjusted close: index = trading day, columns = tickers.

    Share-class tickers are translated from the dotted convention (BRK.B) to the
    dashed one yfinance expects (BRK-B), then renamed back so columns align with the
    signal's ticker index. Fully-empty columns (delisted or unfetchable names) are
    dropped, since they carry no events.
    """
    import yfinance as yf
    tickers = list(dict.fromkeys(tickers))            # de-dup, preserve order
    yf_map = {t: t.replace(".", "-") for t in tickers}
    data = yf.download(list(yf_map.values()), start=start, end=end,
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].rename(columns={v: k for k, v in yf_map.items()})
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close.dropna(how="all").dropna(axis=1, how="all")


def forward_returns(prices, as_of, horizon):
    """Return over `horizon` trading days after `as_of`, per ticker.

    Entry is the first trading day on or after the signal date; exit is `horizon`
    trading days later. Names without enough future data return NaN and are dropped
    downstream, which avoids any look-ahead at the end of the sample.
    """
    idx = prices.index
    pos = idx.searchsorted(pd.Timestamp(as_of))
    if pos >= len(idx) or pos + horizon >= len(idx):
        return pd.Series(dtype=float, index=prices.columns)
    entry = prices.iloc[pos]
    exit_ = prices.iloc[pos + horizon]
    return (exit_ / entry) - 1.0
