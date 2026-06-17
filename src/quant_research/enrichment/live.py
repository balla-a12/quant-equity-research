"""Live enrichment from public sources.

- Committee membership: the @unitedstates/congress-legislators project, fetched and
  cached locally. Members are matched from a Quiver representative name to a bioguide
  id, then to their current committee assignments.
- Net worth: a curated reference table (reference_data/networth.csv), with a small
  embedded fallback.
- Ticker sector: a static map by default (no network), with an opt-in yfinance
  fallback. The static path keeps the backtest fast and free of rate limits; the
  fallback is available when fuller coverage is worth the API cost.

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

# static ticker -> sector, covering the large/mid caps Congress trades most. Sector
# labels match SECTOR_RULES outputs. Extend freely; no network needed.
TICKER_SECTOR = {
    # Technology + communications (folded into Technology)
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "CRM": "Technology", "ADBE": "Technology", "CSCO": "Technology", "INTC": "Technology",
    "AMD": "Technology", "QCOM": "Technology", "TXN": "Technology", "IBM": "Technology",
    "NOW": "Technology", "INTU": "Technology", "AMAT": "Technology", "MU": "Technology",
    "PANW": "Technology", "PLTR": "Technology", "NFLX": "Technology", "TMUS": "Technology",
    "CMCSA": "Technology", "T": "Technology", "VZ": "Technology",
    # Defense / aerospace
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "GD": "Defense", "LHX": "Defense",
    "AXON": "Defense", "BA": "Defense", "HII": "Defense",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "C": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "AXP": "Financials", "V": "Financials", "MA": "Financials", "COF": "Financials",
    "USB": "Financials", "PNC": "Financials", "BX": "Financials", "SPGI": "Financials",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care", "PFE": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "TMO": "Health Care", "ABT": "Health Care",
    "DHR": "Health Care", "BMY": "Health Care", "AMGN": "Health Care", "GILD": "Health Care",
    "CVS": "Health Care", "MDT": "Health Care", "ISRG": "Health Care", "VRTX": "Health Care",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "OXY": "Energy", "PSX": "Energy", "MPC": "Energy", "VLO": "Energy", "KMI": "Energy",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "EXC": "Utilities", "SRE": "Utilities",
    # Consumer Staples
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "MO": "Consumer Staples",
    "PM": "Consumer Staples", "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    "TGT": "Consumer Staples", "CELH": "Consumer Staples",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "DIS": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "MELI": "Consumer Discretionary", "CAVA": "Consumer Discretionary",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials", "DE": "Industrials",
    "UNP": "Industrials", "UPS": "Industrials", "MMM": "Industrials", "EMR": "Industrials",
    "ETN": "Industrials", "ITW": "Industrials", "CSX": "Industrials", "NSC": "Industrials",
}

_YF_SECTOR = {
    "Technology": "Technology", "Industrials": "Industrials",
    "Financial Services": "Financials", "Healthcare": "Health Care",
    "Energy": "Energy", "Utilities": "Utilities",
    "Consumer Defensive": "Consumer Staples", "Consumer Cyclical": "Consumer Discretionary",
    "Communication Services": "Technology",
}

_NETWORTH_FALLBACK = {
    "P000197": 120_000_000, "T000278": 6_000_000, "K000389": 30_000_000,
    "M001157": 100_000_000, "C001120": 3_000_000,
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
    def __init__(self, cache_dir=DEFAULT_CACHE, networth_csv=NETWORTH_CSV,
                 use_yf_sectors=False):
        self.cache_dir = cache_dir
        self.networth_csv = networth_csv
        self.use_yf_sectors = use_yf_sectors    # opt-in network sector lookups
        self._name_to_bio = {}
        self._bio_committees = {}
        self._networth = {}
        self._sector_cache = {}
        self._load()

    def _fetch_yaml(self, filename):
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
            if len(cid) == 4:
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
        if not self.use_yf_sectors:
            return None
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]
        sector = None
        try:
            import yfinance as yf
            info = yf.Ticker(ticker.replace(".", "-")).info
            sector = _YF_SECTOR.get(info.get("sector"))
        except Exception:
            sector = None
        self._sector_cache[ticker] = sector
        return sector

    def prime_sectors(self, tickers):
        for t in tickers:
            self.sector_of(t)
        return sum(1 for t in tickers if self.sector_of(t) is not None)

    def is_aligned(self, representative, ticker):
        sec = self.sector_of(ticker)
        return sec is not None and sec in self.committee_sectors(representative)
