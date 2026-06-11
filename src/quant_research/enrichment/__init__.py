"""Member and security enrichment used by the signals.

Two implementations behind one interface:
  MockEnrichment  - synthetic, offline, deterministic (for mock-data runs)
  LiveEnrichment  - real public sources (committees, net worth, sectors)
"""
