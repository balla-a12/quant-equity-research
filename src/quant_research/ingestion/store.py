"""A thin SQLite storage layer.

We use SQLite because it is a single file with no server to manage, which suits
a Colab/Drive workflow. Writes are idempotent: each row gets a content hash as
its identity, so re-running ingestion never creates duplicate rows.
"""
import hashlib
import pandas as pd
from sqlalchemy import create_engine


def get_engine(db_path="data/quiver.db"):
    return create_engine(f"sqlite:///{db_path}")


def _row_id(row):
    raw = "|".join(str(v) for v in row.values)
    return hashlib.md5(raw.encode()).hexdigest()


def store(df, table, engine):
    """Write df to `table`, de-duplicating against whatever is already there.

    Returns the number of rows in the incoming df. Because the identity of a
    row is a hash of its contents, an identical filing seen twice collapses to
    one stored row.
    """
    df = df.copy()
    df["row_id"] = df.apply(_row_id, axis=1)

    try:
        existing = pd.read_sql(f"SELECT * FROM {table}", engine)
        combined = pd.concat([existing, df], ignore_index=True)
    except Exception:
        combined = df

    combined = combined.drop_duplicates(subset="row_id", keep="last")
    combined.to_sql(table, engine, if_exists="replace", index=False)
    return len(df)


def load(table, engine):
    return pd.read_sql(f"SELECT * FROM {table}", engine)


def table_counts(engine):
    """Quick health check: row count for each known table."""
    counts = {}
    for t in ["congress_trades", "insider_trades", "gov_contracts", "lobbying"]:
        try:
            counts[t] = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t}", engine)["n"].iloc[0]
        except Exception:
            counts[t] = 0
    return counts
