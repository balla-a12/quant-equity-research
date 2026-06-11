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
    def __init__(self, token=None, mock=False):
        # No token means we cannot reach the live API, so we fall back to mock.
        self.mock = mock or token is None
        self._api = None
        if not self.mock:
            import quiverquant            # imported lazily; unused in mock mode
            self._api = quiverquant.quiver(token)

    # ---- Congressional trades -------------------------------------------
    def congress_trades(self):
        raw = (mock_data.mock_congress_trading() if self.mock
               else self._api.congress_trading())
        return self._normalize_congress(raw)

    def _normalize_congress(self, df):
        df = df.copy()
        lows, highs = zip(*df["Range"].map(_parse_range)) if len(df) else ([], [])
        out = pd.DataFrame({
            "ticker": df["Ticker"].str.upper(),
            "transaction_date": pd.to_datetime(df["TransactionDate"]),
            "report_date": pd.to_datetime(df["ReportDate"]),
            "representative": df["Representative"],
            "party": df["Party"],
            "chamber": df["House"],
            "transaction_type": df["Transaction"],
            "amount_min": list(lows),
            "amount_max": list(highs),
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
        raw = (mock_data.mock_gov_contracts() if self.mock
               else self._api.gov_contracts())
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
        raw = (mock_data.mock_lobbying() if self.mock
               else self._api.lobbying())
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
