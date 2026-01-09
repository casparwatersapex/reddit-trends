from __future__ import annotations

from typing import Any

import pandas as pd


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename input columns to canonical names."""
    return df.rename(columns=mapping).copy()


def parse_types(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out
