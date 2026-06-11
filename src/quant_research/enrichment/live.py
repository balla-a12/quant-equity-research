"""Live enrichment from public sources.

- Committee membership: the @unitedstates/congress-legislators project, fetched and
  cached locally. Members are matched from a Quiver representative name to a bioguide
  id, then to their current committee assignments.
- Net worth: a curated reference table (reference_data/networth.csv), with a small
  embedded fallback. Figures are approximate public estimates and are meant to be
  expanded from primary financial-disclosure data over time.
- Ticker sector: a static map with an optional yfinance fallback.

Unmatched members and unmapped tickers degrade silently — the feature simply does
not fire, which never penalizes a name, it only declines to boost it.
"""
import os
import re
import csv
import json
import unicodedata

US_BASE = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/"
DEFAULT_CACHE = "data/reference"
NETWORTH_CSV = "reference_data/networth.csv"

# committee-name keyword -> sectors it has oversight relevance to
SECTOR_RULES = [
    ("armed services", {"Defense"}),
    ("homeland security", {"Defense"}),
    ("foreign affairs", {"Defense"}),
    ("financial services", {"Financials"}),
    ("banking", {"Financials"}),
    ("ways and means", {"Financials"}),
    ("energy", {"Energy", "Utilities"}),
    ("natural resources", {"Energy"}),
    ("commerce", {"Technology", "Consumer Discretionary"}),
    ("science", {"Technology"}),
    ("agriculture", {"Consumer Staples"}),
    ("health", {"Health Care"}),
    ("transportation", {"Industrials"}),
    ("infrastructure", {"Industrials"}),
]

# static ticker -> sector; extend freely. Sectors match SECTOR_RULES outputs.
TICKER_SECTOR = {
    "PLTR": "Technology", "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology",
    "GOOGL": "Technology", "AMZN": "Consumer Discretionary", "META": "Technology",
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "AXON": "Defense",
    "BA": "Defense", "GD": "Defense", "LHX": "Defense",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "NEE": "Utilities", "DUK": "Utilities",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "UNH": "Health Care", "JNJ": "Health Care", "PFE": "Health Care", "LLY": "Health Care",
    "CELH": "Consumer Staples", "KO": "Consumer Staples", "PG": "Consumer Staples",
    "MELI": "Consumer Discretionary", "CAVA": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials", "DE": "Industrials",
}

# yfinance sector strings -> our sector labels
_YF_SECTOR = {
    "Technology": "Technology", "Industrials": "Industrials",
    "Financial Services": "Financials", "Healthcare": "Health Care",
    "Energy": "Energy", "Utilities": "Utilities",
    "Consumer Defensive": "Consumer Staples", "Consumer Cyclical": "Consumer Discretionary",
}

# approximate public net-worth estimates (USD), keyed by bioguide; clearly rough,
# meant as a fallback when reference_data/networth.csv is absent. Expand from
# primary financial-disclosure data for production use.
_NETWORTH_FALLBACK = {
    "P000197": 120_000_000,   # Pelosi
    "T000278": 6_000_000,     # Tuberville
    "K000389": 30_000_000,    # Khanna
    "M001157": 100_000_000,   # McCaul
    "C001120": 3_000_000,     # Crenshaw
}

_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def _firstlast(s):
    toks = [t for t in _norm(s).split() if t not in _SUFFIX]
    return f"{toks[0]} {toks[-1]}" if len(toks) >= 2 else _norm(s)


def _committee_sectors(committee_name):
    out, low = set(), committee_name.lower()
    for kw, secs in SECTOR_RULES:
        if kw in low:
            out |= secs
    return out


class LiveEnrichment:
    def __init__(self, cache_dir=DEFAULT_CACHE, networth_csv=NETWORTH_CSV):
        self.cache_dir = cache_dir
        self.networth_csv = networth_csv
        self._name_to_bio = {}
        self._bio_committees = {}
        self._networth = {}
        self._load()

    # ---- data loading ----------------------------------------------------
    def _fetch_yaml(self, filename):
        """Fetch a congress-legislators YAML, caching the parsed JSON locally."""
        import requests
        import yaml
        os.makedirs(self.cache_dir, exist_ok=True)
        cache = os.path.join(self.cache_dir, filename.replace(".yaml", ".json"))
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)
        data = yaml.safe_load(requests.get(US_BASE + filename, timeout=120).text)
        with open(cache, "w") as f:
            json.dump(data, f)
        return data

    def _load(self):
        legislators = self._fetch_yaml("legislators-current.yaml")
        committees = self._fetch_yaml("committees-current.yaml")
        membership = self._fetch_yaml("committee-membership-current.yaml")

        for p in legislators:
            n, bio = p["name"], p["id"]["bioguide"]
            variants = {f"{n.get('first','')} {n.get('last','')}", n.get("official_full", "")}
            if n.get("nickname"):
                variants.add(f"{n['nickname']} {n.get('last','')}")
            for v in variants:
                if v.strip():
                    self._name_to_bio[_norm(v)] = bio
                    self._name_to_bio[_firstlast(v)] = bio

        comm_name = {c["thomas_id"]: c["name"] for c in committees}
        for cid, members in membership.items():
            if len(cid) == 4:  # top-level committees, not subcommittees
                name = comm_name.get(cid, cid)
                for m in members:
                    self._bio_committees.setdefault(m["bioguide"], set()).add(name)

        self._networth = self._load_networth()

    def _load_networth(self):
        if os.path.exists(self.networth_csv):
            out = {}
            with open(self.networth_csv) as f:
                for row in csv.DictReader(f):
                    bio = row.get("bioguide", "").strip()
                    val = row.get("net_worth", "").strip()
                    if bio and val:
                        out[bio] = float(val)
            if out:
                return out
        return dict(_NETWORTH_FALLBACK)

    # ---- lookups ---------------------------------------------------------
    def resolve(self, representative):
        return (self._name_to_bio.get(_norm(representative))
                or self._name_to_bio.get(_firstlast(representative)))

    def net_worth(self, representative):
        return self._networth.get(self.resolve(representative))

    def committee_sectors(self, representative):
        comms = self._bio_committees.get(self.resolve(representative), set())
        out = set()
        for c in comms:
            out |= _committee_sectors(c)
        return out

    def sector_of(self, ticker):
        if ticker in TICKER_SECTOR:
            return TICKER_SECTOR[ticker]
        try:
            import yfinance as yf
            return _YF_SECTOR.get(yf.Ticker(ticker).info.get("sector"))
        except Exception:
            return None

    def is_aligned(self, representative, ticker):
        sec = self.sector_of(ticker)
        return sec is not None and sec in self.committee_sectors(representative)
