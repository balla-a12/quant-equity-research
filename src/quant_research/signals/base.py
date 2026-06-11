"""The signal contract every signal implements.

A signal turns raw alternative data into a per-ticker score on a 0-100 scale, so
that scores from different datasets are directly comparable and can be blended
into a composite. Keeping every signal behind one small interface means the
backtester and the composite scorer depend on the contract, never on the details
of any single dataset.
"""
from abc import ABC, abstractmethod
from datetime import date
import pandas as pd


class BaseSignal(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def compute(self, as_of: date | None = None) -> pd.DataFrame:
        """Return a DataFrame indexed by ticker with a 'score' column in [0, 100]."""
        ...

    @staticmethod
    def _scale_0_100(s: pd.Series) -> pd.Series:
        """Min-max a series onto [0, 100]; a flat series maps to zeros."""
        if s.empty or s.max() == s.min():
            return pd.Series(0.0, index=s.index)
        return (s - s.min()) / (s.max() - s.min()) * 100.0
