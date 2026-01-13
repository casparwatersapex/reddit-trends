from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_tabular(path: str | Path) -> pd.DataFrame:
    """Read a CSV, Parquet, or JSONL file into a DataFrame.

    Keep this function narrow and predictable. If you add formats (xlsx), do it explicitly.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {suffix}. Supported: .csv, .parquet, .jsonl")
