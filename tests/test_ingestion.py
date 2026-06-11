"""Tests for the ingestion layer: normalization schema and idempotent storage."""
import pandas as pd
from quant_research.ingestion.client import QuiverClient, _parse_range
from quant_research.ingestion import store


def test_parse_range():
    assert _parse_range("$15,001 - $50,000") == (15001, 50000)
    assert _parse_range("$1,001 - $15,000") == (1001, 15000)


def test_congress_schema():
    df = QuiverClient(mock=True).congress_trades()
    expected = {"ticker", "transaction_date", "representative", "party",
                "chamber", "transaction_type", "amount_min", "amount_max"}
    assert expected.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["transaction_date"])
    assert (df["amount_max"] >= df["amount_min"]).all()


def test_insider_value_computed():
    df = QuiverClient(mock=True).insider_trades()
    assert "value" in df.columns
    assert (df["value"] == (df["shares"] * df["price_per_share"]).round(2)).all()


def test_storage_idempotent(tmp_path):
    client = QuiverClient(mock=True)
    engine = store.get_engine(f"{tmp_path}/t.db")
    store.store(client.congress_trades(), "congress_trades", engine)
    first = store.table_counts(engine)["congress_trades"]
    store.store(client.congress_trades(), "congress_trades", engine)
    second = store.table_counts(engine)["congress_trades"]
    assert first == second
