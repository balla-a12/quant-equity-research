"""Synthetic enrichment: self-contained, offline, deterministic.

Used when running on mock trades without reaching any external source. The maps
mirror the shape of the live data so signal behavior is comparable across modes.
"""

TICKER_SECTOR = {
    "PLTR": "Technology", "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology",
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "AXON": "Defense", "BA": "Defense",
    "CELH": "Consumer Staples", "MELI": "Consumer Discretionary", "CAVA": "Consumer Discretionary",
    "GE": "Industrials", "CAT": "Industrials", "JPM": "Financials",
}
COMMITTEE_SECTORS = {
    "Armed Services": {"Defense"},
    "Science, Space, and Technology": {"Technology"},
    "Financial Services": {"Financials"},
    "Energy and Commerce": {"Energy", "Technology"},
    "Agriculture": {"Consumer Staples"},
    "Transportation and Infrastructure": {"Industrials"},
}
REP_COMMITTEE = {
    "Nancy Pelosi": "Science, Space, and Technology", "Dan Crenshaw": "Armed Services",
    "Josh Gottheimer": "Financial Services", "Marjorie Taylor Greene": "Armed Services",
    "Ro Khanna": "Armed Services", "Tommy Tuberville": "Armed Services",
    "Michael McCaul": "Armed Services", "Virginia Foxx": "Science, Space, and Technology",
    "Earl Blumenauer": "Transportation and Infrastructure", "Kathy Manning": "Financial Services",
}
REP_NETWORTH = {
    "Nancy Pelosi": 250_000_000, "Dan Crenshaw": 3_000_000, "Josh Gottheimer": 8_000_000,
    "Marjorie Taylor Greene": 11_000_000, "Ro Khanna": 60_000_000, "Tommy Tuberville": 6_000_000,
    "Michael McCaul": 125_000_000, "Virginia Foxx": 9_000_000, "Earl Blumenauer": 2_000_000,
    "Kathy Manning": 40_000_000,
}

SECTOR_REPS = {}
for _rep, _comm in REP_COMMITTEE.items():
    for _sec in COMMITTEE_SECTORS.get(_comm, set()):
        SECTOR_REPS.setdefault(_sec, []).append(_rep)


class MockEnrichment:
    def net_worth(self, representative):
        return REP_NETWORTH.get(representative)

    def committee_sectors(self, representative):
        return COMMITTEE_SECTORS.get(REP_COMMITTEE.get(representative), set())

    def sector_of(self, ticker):
        return TICKER_SECTOR.get(ticker)

    def is_aligned(self, representative, ticker):
        sec = self.sector_of(ticker)
        return sec is not None and sec in self.committee_sectors(representative)
