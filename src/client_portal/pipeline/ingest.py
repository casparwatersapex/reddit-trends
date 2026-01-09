from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def read_tabular(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Keep this function narrow and predictable. If you add formats (xlsx), do it explicitly.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}. Supported: .csv, .parquet")
