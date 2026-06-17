"""A unified client for Quiver Quantitative data.

The QuiverClient exposes one method per dataset. Each method fetches raw data
(from the live API when a token is supplied, otherwise from the mock generator)
and returns it normalized into a consistent internal schema. The rest of the
project depends only on that internal schema, so nothing downstream needs to
know or care whether the data came from the live API or the mock source.
"""
import re
import pandas as pd

from . import mock_data


def _parse_range(text):
    """Turn a Quiver amount range like '$15,001 - $50,000' into (low, high)."""
    nums = re.findall(r"[\d,]+", str(text))
    vals = [int(n.replace(",", "")) for n in nums if n.replace(",", "").isdigit()]
    if len(vals) >= 2:
        return vals[0], vals[1]
    if len(vals) == 1:
        return vals[0], vals[0]
    return 0, 0


class QuiverClient:
    def __init__(self, token=None, mock=False, mock_history_days=40):
        self.mock_history_days = mock_history_days
        # No token means we cannot reach the live API, so we fall back to mock.
        self.mock = mock or token is None
        self._api = None
        if not self.mock:
            import quiverquant            # imported lazily; unused in mock mode
            self._api = quiverquant.quiver(token)

    # ---- Congressional trades -------------------------------------------
    def congress_trades(self, historical=False):
        if self.mock:
            raw = mock_data.mock_congress_trading(
                history_days=self.mock_history_days,
                n=max(180, self.mock_history_days * 4))
        else:
            raw = self._api.congress_trading(recent=not historical)
        return self._normalize_congress(raw)

    @staticmethod
    def _col(df, *names, default=""):
        """First present column among `names`, else a default-filled Series.

        The recent and bulk Quiver endpoints differ in their column names, so
        every field is looked up tolerantly rather than assumed.
        """
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([default] * len(df), index=df.index)

    def _normalize_congress(self, df):
        df = df.copy()
        if "Range" in df.columns:
            lows, highs = zip(*df["Range"].map(_parse_range)) if len(df) else ([], [])
            amount_min, amount_max = list(lows), list(highs)
        else:
            amt = (self._col(df, "Amount", "Trade_Size_USD", "amount", default=0)
                   .astype(str).str.replace(r"[$,]", "", regex=True))
            amt = pd.to_numeric(amt, errors="coerce").fillna(0)
            amount_min, amount_max = amt.tolist(), amt.tolist()
        out = pd.DataFrame({
            "ticker": self._col(df, "Ticker").astype(str).str.upper(),
            "transaction_date": pd.to_datetime(self._col(df, "TransactionDate", "Traded"),
                                               errors="coerce"),
            "report_date": pd.to_datetime(self._col(df, "ReportDate", "Filed"),
                                          errors="coerce"),
            "representative": self._col(df, "Representative", "Name"),
            "party": self._col(df, "Party"),
            "chamber": self._col(df, "House", "Chamber"),
            "transaction_type": self._col(df, "Transaction").astype(str).str.strip().str.title(),
            "amount_min": amount_min,
            "amount_max": amount_max,
        })
        return out

    # ---- Insider transactions -------------------------------------------
    def insider_trades(self):
        raw = (mock_data.mock_insiders() if self.mock
               else self._api.insiders())
        return self._normalize_insiders(raw)

    def _normalize_insiders(self, df):
        df = df.copy()
        code_map = {"P": "Purchase", "S": "Sale"}
        out = pd.DataFrame({
            "ticker": df["Ticker"].str.upper(),
            "transaction_date": pd.to_datetime(df["Date"]),
            "insider_name": df["Name"],
            "title": df["Title"],
            "transaction_type": df["TransactionCode"].map(code_map).fillna("Other"),
            "shares": df["Shares"].astype(int),
            "price_per_share": df["PricePerShare"].astype(float),
        })
        out["value"] = (out["shares"] * out["price_per_share"]).round(2)
        return out

    # ---- Government contracts --------------------------------------------
    def gov_contracts(self):
        if self.mock:
            days = max(self.mock_history_days, 120)
            raw = mock_data.mock_gov_contracts(history_days=days, n=max(80, days * 2))
        else:
            raw = self._api.gov_contracts()
        return self._normalize_gov(raw)

    def _normalize_gov(self, df):
        df = df.copy()
        out = pd.DataFrame({
            "ticker": df["Ticker"].str.upper(),
            "award_date": pd.to_datetime(df["Date"]),
            "amount": df["Amount"].astype(float),
            "agency": df["Agency"],
        })
        return out

    # ---- Lobbying --------------------------------------------------------
    def lobbying(self):
        if self.mock:
            days = max(self.mock_history_days, 360)
            raw = mock_data.mock_lobbying(history_days=days, n=max(140, days))
        else:
            raw = self._api.lobbying()
        return self._normalize_lobbying(raw)

    def _normalize_lobbying(self, df):
        df = df.copy()
        out = pd.DataFrame({
            "ticker": df["Ticker"].str.upper(),
            "filing_date": pd.to_datetime(df["Date"]),
            "amount": df["Amount"].astype(float),
            "client": df["Client"],
            "issue": df["Issue"],
        })
        return out
