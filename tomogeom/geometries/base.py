from abc import ABC, abstractmethod
import pandas as pd


class _Geometry(ABC):
    """Base class defining geometry interface."""

    @property
    @abstractmethod
    def particles(self) -> pd.DataFrame:
        """Implementations should return a cryopose dataframe."""
        pass
